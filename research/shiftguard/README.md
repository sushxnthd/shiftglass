# ShiftGuard EMBER2024 temporal experiment

This branch contains a kill-or-promote experiment for a distributionally robust learned Bloom filter under temporal query shift.

## Data and chronology

The script downloads the official **EMBER2024 PDF** train/test archives. It uses:

- weeks 0–39: fit a compact 256-feature linear maliciousness oracle;
- weeks 40–45: define the reference non-key score distribution;
- weeks 46–51: select the robustness radius from recent historical drift;
- official weeks 52–63: untouched future test weeks.

The claimed operating radius is therefore fixed before any future test result is observed. A full rho sweep is reported only as a descriptive robustness-memory phase diagram.

Malicious files from weeks 0–51 are treated as the membership keys. Benign files are non-key queries.

## Methods

For the same score regions, the experiment compares:

1. classical Bloom filter at the **1% operational FPR limit**;
2. ordinary partitioned learned allocation at the same 1% limit;
3. ShiftGuard allocation at a precommitted **0.6% design FPR**, minimizing Bloom-filter memory subject to an exact worst-case FPR constraint over a total-variation ball around the reference query distribution.

The 0.6% design target intentionally reserves statistical headroom below the 1% operational limit. With a one-sided 95% Clopper-Pearson deployment audit and 2,000 verified non-key queries, a true 0.6% FPR has about 58% probability of being certified below 1%; using a 1% design target would make a 50% audit-acceptance requirement statistically incoherent.

The inner TV maximization is solved exactly by moving probability mass from the lowest-FPR region to the highest-FPR region. The outer allocation is solved numerically. A flat allocation at the target FPR is always feasible and is used as a safe fallback.

The experiment counts the deployed linear oracle parameters in end-to-end memory. It also simulates a direct deploy-time certificate using a one-sided exact Clopper-Pearson upper bound over sampled verified non-key queries.

## Precommitted kill criteria

Promote the idea only if the **single operating point whose radius was selected from weeks 46–51, before future evaluation**, simultaneously achieves:

- >= 80% coverage of the 12 future EMBER weeks by its declared TV robustness radius;
- >= 15% **end-to-end** memory saving versus a classical Bloom filter at the 1% operational limit, including oracle parameters;
- worst-case FPR <= the 0.6% design target throughout the declared robustness ball;
- zero ShiftGuard future-week expected-FPR breaches of the 1% operational limit;
- at 2,000 audited non-key queries, >= 50% certificate acceptance across temporal test workloads;
- <= 5% empirical false-certification rate in the Monte Carlo audit;
- for a full `PROMOTE`, the ordinary 1%-target PLBF must breach 1% on at least one untouched future week, demonstrating that the real temporal workload actually exposes the failure ShiftGuard is designed to fix.

If all ShiftGuard core criteria pass but the ordinary PLBF never breaches 1% on this PDF slice, the result is `REPLICATE_STRONGER_SHIFT`, not a claimed win: the method survived, but this benchmark did not demonstrate the real failure mode.

If the preselected operating point fails the core criteria, the result is `KILL_OR_REDESIGN`; the experiment is not to be re-labelled as a success.

## Outputs

The workflow uploads and commits no generated data. Its artifact contains summary.json, rho_sweep.csv, weekly_temporal.csv, and audit_certificate.csv.

The raw EMBER archives remain external and are not redistributed.
