from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

import config
from scils_tc.detrending.gwl_lookup import format_target_label, get_filename_suffix, resolve_target

PLACEHOLDER_EVENT_ID = -1


@dataclass(frozen=True)
class TargetSpec:
    """Represent one resolved detrending target in year and GWL space."""

    effective_gwl: float
    effective_year: float
    specified_by_year: bool

    @classmethod
    def from_inputs(cls, target_year=None, target_gwl=None, default_gwl=None) -> "TargetSpec":
        """Resolve CLI target inputs into a normalized target specification."""
        effective_gwl, effective_year, specified_by_year = resolve_target(
            target_year=target_year,
            target_gwl=target_gwl,
            default_gwl=default_gwl,
        )
        return cls(
            effective_gwl=effective_gwl,
            effective_year=effective_year,
            specified_by_year=specified_by_year,
        )

    @property
    def suffix(self) -> str:
        """Return the canonical filename suffix for this target."""
        return get_filename_suffix(
            self.effective_gwl,
            self.effective_year,
            self.specified_by_year,
        )

    @property
    def label(self) -> str:
        """Return the human-readable label for this target."""
        return format_target_label(
            self.effective_gwl,
            self.effective_year,
            self.specified_by_year,
        )


@dataclass(frozen=True)
class SimulationArtifact:
    """Build canonical filenames for simulation CSVs and validation plots."""

    region: str
    n_iter: int | None
    data_mode: str | None = None
    target_spec: TargetSpec | None = None

    @classmethod
    def from_mode(cls, mode: str, region: str, n_iter: int | None) -> "SimulationArtifact":
        """Create a simulation artifact from the internal mode string."""
        if mode == "historical":
            return cls(region=region, n_iter=n_iter, data_mode="historical")

        if mode.startswith("pi_only_"):
            return cls(
                region=region,
                n_iter=n_iter,
                data_mode="pi_only",
                target_spec=_target_spec_from_mode(mode.removeprefix("pi_only_")),
            )

        if mode.startswith("cgi_only_"):
            return cls(
                region=region,
                n_iter=n_iter,
                data_mode="cgi_only",
                target_spec=_target_spec_from_mode(mode.removeprefix("cgi_only_")),
            )

        return cls(
            region=region,
            n_iter=n_iter,
            data_mode=None,
            target_spec=_target_spec_from_mode(mode),
        )

    def simulation_filename(self) -> str:
        """Return the canonical simulation CSV filename."""
        region_str = f"_{self.region}"
        n_iter_str = f"_n{self.n_iter}" if self.n_iter is not None else ""

        if self.data_mode == "historical":
            stem = "simulated_events_historical"
        elif self.data_mode in {"pi_only", "cgi_only"}:
            stem = f"simulated_events_{self.data_mode}_{self.require_target().suffix}"
        elif self.target_spec is not None:
            stem = f"simulated_events_{self.target_spec.suffix}"
        else:
            stem = "simulated_events"

        return f"{stem}{region_str}{n_iter_str}.csv"

    def validation_plot_filename(self) -> str:
        """Return the canonical validation plot filename."""
        region_str = f"_{self.region}"
        n_iter_str = f"_n{self.n_iter}" if self.n_iter is not None else ""

        if self.data_mode == "historical":
            stem = "simulation_validation_historical"
        elif self.data_mode in {"pi_only", "cgi_only"}:
            stem = f"simulation_validation_{self.data_mode}_{self.require_target().suffix}"
        elif self.target_spec is not None:
            stem = f"simulation_validation_{self.target_spec.suffix}"
        else:
            stem = "simulation_validation"

        return f"{stem}{region_str}{n_iter_str}.png"

    def simulation_path(self, base_dir: Path | None = None) -> Path:
        """Return the full path to the canonical simulation CSV."""
        if base_dir is None:
            base_dir = config.SIMULATED_DIR
        return Path(base_dir) / self.simulation_filename()

    def require_target(self) -> TargetSpec:
        """Return the target spec or raise when the artifact is historical."""
        if self.target_spec is None:
            raise ValueError("SimulationArtifact target_spec is required for this data mode")
        return self.target_spec


@dataclass(frozen=True)
class SimulationEnsemble:
    """Represent the simulated year and iteration grid for one ensemble."""

    start_year: int
    end_year: int
    n_iter: int

    @property
    def years(self) -> list[int]:
        """Return the list of simulated years."""
        return list(range(self.start_year, self.end_year + 1))

    @property
    def iterations(self) -> list[int]:
        """Return the list of iteration indices."""
        return list(range(self.n_iter))

    def year_iteration_frame(self, years: list[int] | None = None) -> pd.DataFrame:
        """Return the full year-iteration grid as a DataFrame."""
        years_to_use = self.years if years is None else list(years)
        return pd.DataFrame(product(years_to_use, self.iterations), columns=["year", "iteration"])

    def attach_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach ensemble metadata needed by downstream sparse-event consumers."""
        df = df.copy()
        if "is_placeholder" not in df.columns:
            df["is_placeholder"] = False
        df["simulation_start_year"] = self.start_year
        df["simulation_end_year"] = self.end_year
        df["simulation_n_iter"] = self.n_iter
        return df

    def placeholder_event(
        self,
        *,
        year: int,
        iteration: int,
        enso_state: str,
        mdr_sst=None,
        roni_anomaly=None,
        scaled_mdr_cgi=None,
    ) -> dict:
        """Create a placeholder event for a year-iteration with zero simulated storms."""
        return {
            "year": year,
            "iteration": iteration,
            "event_id": PLACEHOLDER_EVENT_ID,
            "month": pd.NA,
            "lmi_lon": pd.NA,
            "lmi_lat": pd.NA,
            "pi": pd.NA,
            "lmi_pi_ratio": pd.NA,
            "lmi": pd.NA,
            "lmi_sshs": pd.NA,
            "region_name": pd.NA,
            "region_number": pd.NA,
            "lfi_lmi_ratio": pd.NA,
            "lfi": pd.NA,
            "lfi_sshs": pd.NA,
            "mdr_sst_degc": mdr_sst,
            "scaled_mdr_cgi": scaled_mdr_cgi,
            "enso_state": enso_state,
            "roni_anomaly": roni_anomaly,
            "is_placeholder": True,
        }


def count_actual_events(df: pd.DataFrame) -> int:
    """Count non-placeholder simulation rows."""
    if "is_placeholder" not in df.columns:
        return int(len(df))
    return int((~df["is_placeholder"].fillna(False)).sum())


def actual_event_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only non-placeholder simulation rows."""
    if "is_placeholder" not in df.columns:
        return df
    return df.loc[~df["is_placeholder"].fillna(False)].copy()


def year_iteration_grid_from_dataframe(df: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """Rebuild the full year-iteration grid from simulation metadata when available."""
    if len(df) == 0:
        return pd.DataFrame(columns=["year", "iteration"])

    if years is None:
        if {"simulation_start_year", "simulation_end_year"}.issubset(df.columns):
            start_year = int(df["simulation_start_year"].dropna().iloc[0])
            end_year = int(df["simulation_end_year"].dropna().iloc[0])
            years_to_use = list(range(start_year, end_year + 1))
        else:
            years_to_use = sorted(int(year) for year in df["year"].dropna().unique())
    else:
        years_to_use = [int(year) for year in years]

    if "simulation_n_iter" in df.columns and not df["simulation_n_iter"].dropna().empty:
        n_iter = int(df["simulation_n_iter"].dropna().iloc[0])
        iterations = list(range(n_iter))
    else:
        iterations = sorted(int(iteration) for iteration in df["iteration"].dropna().unique())

    return pd.DataFrame(product(years_to_use, iterations), columns=["year", "iteration"])


def _target_spec_from_mode(mode: str) -> TargetSpec:
    """Convert an internal mode string into a normalized target specification."""
    if mode.startswith("year_"):
        return TargetSpec.from_inputs(target_year=int(mode.removeprefix("year_")))
    if mode.startswith("to_"):
        return TargetSpec.from_inputs(target_year=int(mode.removeprefix("to_")))
    if mode.startswith("GWL_"):
        return TargetSpec.from_inputs(target_gwl=float(mode.removeprefix("GWL_")))
    if mode.startswith("GWL"):
        return TargetSpec.from_inputs(target_gwl=float(mode.removeprefix("GWL")))
    raise ValueError(f"Unknown simulation mode: {mode}")