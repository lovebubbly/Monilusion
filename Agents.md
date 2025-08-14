@"

# Repository Guidelines

## Project Structure & Module Organization

- `src/`: Primary Python modules/packages for reusable code.
- `intelligence_package/`: Domain-specific utilities and algorithms.
- `legacy/`: Archived/older experiments; do not extend without review.
- `data/`, `saved_models/`, `cache/`: Large artifacts and outputs; not source.
- `real_M1.py`: Top-level runnable script/entry point.
- `.env`: Local configuration; never commit secrets.

## Build, Test, and Development Commands

- Activate env: `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows).
- Run main script: `python real_M1.py`
- Run a module: `python -m src.<module_path>`
- Lint (if installed): `ruff check src intelligence_package` or `flake8 src intelligence_package`
- Format (if installed): `black src intelligence_package`
- Dependencies: list/pin in `requirements.txt` if formalizing dependencies.

## Coding Style & Naming Conventions

- Indentation: 4 spaces; follow PEP 8.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `CONSTANT_CASE` for constants.
- Imports: Prefer absolute package imports from `src`/`intelligence_package`.
- Formatting: Prefer Black (line length 88); keep diffs minimal.
- Linting: Prefer Ruff or Flake8; fix warnings proactively.

## Testing Guidelines

- Framework: Prefer `pytest`; fallback to `unittest` if needed.
- Location: Mirror source under `src/tests/` (e.g., `src/tests/test_module.py`).
- Naming: `test_*.py` files; functions `test_*`.
- Run tests: `pytest -q` or `python -m pytest -q`
- Coverage target: Aim for 80%+ on core logic; exclude `legacy/`, data/model artifacts.

## Commit & Pull Request Guidelines

- Commits: Use Conventional Commits, e.g., `feat: add data loader`, `fix: handle NaN in metrics`, `chore: update .gitignore`.
- Scope small and focused; include rationale in the body if non-trivial.
- PRs: Clear description, linked issues, steps to reproduce/validate, and sample outputs when relevant. Tag reviewers and note any data/model requirements.

## Security & Configuration Tips

- Keep secrets in `.env` only; never commit credentials or API keys.
- Avoid committing large files from `data/`, `saved_models/`, `cache/`; `.gitignore` should cover these—extend if needed.
  "@ | Set-Content -Encoding UTF8 -NoNewline -Path .\AGENTS.md
