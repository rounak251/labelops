# Contributing

Changes should keep LabelOps easy to reproduce and honest about what its experiment measures. Small improvements with a clear hypothesis, test, and limitation are welcome.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
```

Run the benchmark when changing model behavior, acquisition, metrics, or splitting:

```bash
labelops benchmark --output reports
```

Review both the machine-readable results and the rendered report. Record the configuration and environment for any performance comparison. The reference environment was tested with Python 3.13 and constrained dependencies.

## Required boundaries

- Fit preprocessing and models on the currently labeled training rows only.
- Keep validation and test examples out of fitting, review, and acquisition.
- Keep oracle-label simulation confined to the offline benchmark.
- Preserve append-only annotation history and latest-vote consensus semantics.
- Validate labels, sample membership, and reviewer fields at the service boundary.
- Reject persisted annotation records outside the training pool before fitting.
- Keep model provenance aligned with the actual fitting IDs and label sources.
- Validate saved artifacts against the model configuration, dependency versions, and estimator parameters.
- Describe results with their split, seed aggregation, and label budget.

Add meaningful behavior tests when changing these contracts. Avoid tests that merely duplicate the implementation or assert an entire output snapshot without checking the underlying invariant.

## Pull requests

Explain the problem, the resulting behavior, and how you verified it. Include measured before/after results only when the comparison uses the same protocol. Update the model card and documentation when assumptions or interfaces change.

Do not commit local workspaces, serialized personal artifacts, credentials, private reviewer information, or unrelated data. Keep extension ideas distinct from implemented capabilities. Dataset adapters need source credit, license information, and an evaluation split appropriate to the new domain.
