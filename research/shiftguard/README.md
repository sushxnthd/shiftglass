# ShiftGuard EMBER2024 temporal experiment

This branch contains a kill-or-promote experiment for a distributionally robust learned Bloom filter under temporal query shift.

## Data and chronology

The script downloads the official **EMBER2024 PDF** train/test archives. It uses:

- weeks 0–39: fit a compact 256-feature linear maliciousness oracle;
- weeks 40–51: estimate the non-key score distribution used to allocate the learned Bloom filter;
- official weeks 52–63: untouched future test weeks.

Malicious files from weeks 0–51 are treated as the membership keys. Benign files are non-key queries.

## Methods

For the same score regions, the experiment compares:

1. classical Bloom filter at target FPR;
2. ordinary partitioned learned allocation;
3. ShiftGuard allocation minimizing Bloom-filter memory subject to an exact worst-case FPR constraint over a total-variation ball around the calibration query distribution.

The inner TV maximization is solved exactly by moving probability mass from the lowest-FPR region to the highest-FPR region. The outer allocation is solved numerically. A flat allocation at the target FPR is always feasible and is used as a safe fallback.

The experiment counts the deployed linear oracle parameters in end-to-end memory. It also simulates a direct deploy-time certificate using a one-sided exact Clopper-Pearson upper bound over sampled verified non-key queries.

## Precommitted kill criteria

Promote the idea only if one robust operating point simultaneously achieves:

- >= 80% coverage of the 12 future EMBER weeks by its declared TV robustness radius;
- >= 15% **end-to-end** memory saving versus a classical Bloom filter, including oracle parameters;
- worst-case FPR <= the 1% design target throughout the declared robustness ball;
- at 1,000 audited non-key queries, >= 50% certificate acceptance across temporal test workloads;
- <= 5% empirical false-certification rate in the Monte Carlo audit.

If no operating point meets all conditions, the result is KILL_OR_REDESIGN; the experiment is not to be re-labelled as a success.

## Outputs

The workflow uploads and commits no generated data. Its artifact contains summary.json, rho_sweep.csv, weekly_temporal.csv, and audit_certificate.csv.

The raw EMBER archives remain external and are not redistributed.
