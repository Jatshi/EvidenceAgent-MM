# Contributing

Create a feature branch, add a failing test for behavior changes, implement the smallest coherent slice, and run:

```bash
ruff check .
ruff format --check .
pytest --cov=evidenceagent_mm --cov-report=term-missing
```

Do not commit media, databases, weights, credentials, or benchmark numbers without the generating command and environment manifest. Pull requests that change answer behavior must report answered/clarification/abstention counts separately.
