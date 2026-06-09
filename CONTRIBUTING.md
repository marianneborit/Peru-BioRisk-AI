# Contributing to Peru BioRisk AI

Thank you for your interest in contributing! This project aims to be
a community-driven open science tool for public health in Peru and beyond.

## Code of conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
In short: be respectful, inclusive, and assume good faith.

## Ways to contribute

- **Bug reports** — open an issue with a minimal reproducible example
- **Feature requests** — open an issue tagged `enhancement`
- **Data integrations** — new data sources or country adaptations
- **Model improvements** — better architectures, features, or validation
- **Documentation** — tutorials, docstrings, translations
- **Scientific review** — feedback on methodology and assumptions

## Development setup

```bash
git clone https://github.com/peru-biorisk-ai/peru-biorisk-ai.git
cd peru-biorisk-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feature/<short-description>` | `feature/chirps-ingestion` |
| Bug fix | `fix/<short-description>` | `fix/ndvi-null-handling` |
| Documentation | `docs/<topic>` | `docs/api-examples` |
| Data | `data/<source>` | `data/senamhi-v2` |

## Pull request checklist

Before submitting a PR, please confirm:

- [ ] All existing tests pass (`pytest tests/`)
- [ ] New functionality has tests (aim for ≥ 80% coverage on new code)
- [ ] Code passes linting (`ruff check src/ tests/`)
- [ ] Type hints added and mypy passes
- [ ] Docstrings follow Google style for all public functions
- [ ] `CHANGELOG.md` updated
- [ ] PR description explains *what* and *why* (not just *how*)

## Code style

We use `ruff` and `black` (line length 100). Run both before committing:

```bash
ruff check --fix src/ tests/
black src/ tests/
```

## Testing

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_features.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## Scientific methodology

Changes to ML models or feature engineering should include:

1. A brief explanation of the scientific motivation
2. Cross-validation results (spatial block CV with AUC-ROC, AUC-PR, Brier score)
3. Comparison against the current baseline
4. Reference to supporting literature (BibTeX preferred)

## Data licensing

All data contributed to the repository must be compatible with CC-BY 4.0
or more permissive licenses. Do not commit raw data files — add download
scripts to `src/ingestion/` and document the source in `docs/data_dictionary.md`.

## Questions?

Open a [GitHub Discussion](https://github.com/peru-biorisk-ai/peru-biorisk-ai/discussions)
or join our Slack community (link in repository description).
