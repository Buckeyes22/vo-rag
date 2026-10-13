# security tooling

`scripts/ci.sh` runs `detect-secrets` against `.secrets.baseline`, alongside the bandit configuration in `pyproject.toml` under `[tool.bandit]`.
