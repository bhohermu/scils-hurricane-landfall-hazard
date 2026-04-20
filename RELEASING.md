# Releasing

## Versioning plan

- `0.1.0`: initial public repository release during paper typesetting
- `1.0.0`: manuscript-aligned release once the paper and Zenodo archive are public

## Release workflow

1. Update `CHANGELOG.md` and `CITATION.cff`.
2. Confirm that restricted data and generated outputs are not staged.
3. Tag the release in git.
4. Publish the release on GitHub.
5. Archive the GitHub release with Zenodo.
6. Update the repository documentation with the Zenodo DOI and article citation.

## Public-release checks

- `python -m pip install -e . --no-deps`
- `python -m unittest tests.test_public_repo -v`
- `python -m unittest test_model.py -v`
- verify CLI help for all `run_*.py` entrypoints
- verify `.gitignore` excludes raw inputs and generated outputs