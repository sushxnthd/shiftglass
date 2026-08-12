# shiftglass

[![CI](https://img.shields.io/github/actions/workflow/status/sushxnthd/shiftglass/ci.yml?branch=main&label=CI)](.github/workflows/ci.yml) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[**Live demo →**](https://sushxnthd.github.io/shiftglass/) · [Architecture](docs/architecture.md) · [Benchmark](benchmarks/results.json)

`shiftglass` is a dependency-light drift report for numerical ML data. It separates **which columns moved** from **whether the joint distribution moved**.

```bash
pip install -e .
python examples/make_example.py
shiftglass examples/drift.npz
```

## Diagnostics

Per feature:

- two-sample Kolmogorov–Smirnov statistic;
- Jensen–Shannon divergence over matched histograms;
- standardized mean shift;
- standard-deviation ratio.

Global:

- RBF maximum mean discrepancy (MMD²), using a median-distance bandwidth heuristic;
- covariance-matrix movement.

```python
from shiftglass import drift_report

report = drift_report(reference, current, feature_names=columns)
print(report.features[0])
```

## Benchmark

`python benchmarks/run.py` injects mean/variance changes into known dimensions of a five-feature Gaussian dataset. The checked result records the top drifted dimensions and global MMD².

## Design boundary

`shiftglass` detects statistical movement; it does not claim the movement is harmful. Pair drift diagnostics with task performance and domain thresholds before triggering production actions.

## Roadmap

- categorical drift metrics
- rolling-window reports
- bootstrap uncertainty
- online reference-window policies

MIT licensed.
