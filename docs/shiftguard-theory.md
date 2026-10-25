# ShiftGuard theory note

## Setup

Let score space be partitioned into \(k\) regions. Let

- \(g\in\Delta^k\) be the key-mass distribution across regions,
- \(h\in\Delta^k\) be a reference non-key query distribution,
- \(f_i\in(0,1]\) be the false-positive probability assigned to region \(i\),
- \(F\) be the target overall false-positive rate,
- \(\rho\in[0,1]\) be the allowed total-variation shift.

Ignoring the fixed learned-oracle cost, the idealized Bloom-filter memory is

\[
M(f)=\frac{n}{(\ln 2)^2}\sum_i g_i[-\ln f_i].
\]

ShiftGuard solves

\[
\min_f M(f)
\quad\text{s.t.}\quad
\sup_{q\in\Delta^k:\operatorname{TV}(q,h)\le \rho}q^\top f\le F.
\]

The guarantee is conditional on the declared shift set. A deployment audit is used separately to decide whether the live workload can still be certified.

## Proposition 1 — exact finite-region adversary

For fixed \(f\), the worst distribution in a TV ball can be found by moving at most \(\rho\) probability mass from the smallest-\(f_i\) regions to the largest-\(f_i\) regions, respecting the available donor mass and receiver capacity.

This is not a heuristic. It is the solution of

\[
\max_q q^\top f
\quad
\text{s.t. }q\ge0,\ \mathbf 1^\top q=1,\ 
\frac12\lVert q-h\rVert_1\le\rho.
\]

A computational cross-check compares the mass-transfer algorithm against the full linear program on randomized instances.

## Proposition 2 — convex robust allocation

Introduce \(a,b\ge0\), \(c\ge0\), and free scalar \(\lambda\). The LP dual of the inner problem is

\[
\min_{a,b,c,\lambda}
h^\top(a-b)+2\rho c+\lambda
\]

subject to

\[
a-b+\lambda\mathbf 1\ge f,\qquad
c\mathbf 1\ge a+b.
\]

Therefore the robust constraint is equivalent to the existence of dual variables satisfying

\[
h^\top(a-b)+2\rho c+\lambda\le F.
\]

All these constraints are linear in the decision variables. Since
\(-\sum_i g_i\ln f_i\) is convex on \(f_i>0\), the complete fixed-region ShiftGuard problem is a convex program. A global optimum can therefore be obtained with a convex solver; there is no need to rely on a local nonlinear search.

## Proposition 3 — zero-shift limit and information value

At \(\rho=0\), ShiftGuard reduces to the ordinary fixed-region PLBF allocation. If no regional FPR clips at 1 and \(h_i>0\),

\[
f_i^*=F\frac{g_i}{h_i}.
\]

Then

\[
\frac{M(f^*)}{n}
=
\frac{-\ln F-D_{\mathrm{KL}}(g\Vert h)}{(\ln2)^2}.
\]

Thus, in the unclipped regime, the learned filter's memory advantage over a classical Bloom filter is exactly the score-space separation \(D_{\mathrm{KL}}(g\Vert h)\), scaled by \(1/(\ln2)^2\).

This gives ShiftGuard a useful interpretation: increasing robustness radius forces the allocation to give back some of the memory gain purchased from workload-specific information.

## Proposition 4 — classical-filter collapse

If

\[
\rho\ge 1-\min_i h_i,
\]

then the TV ball around \(h\) contains every simplex vertex \(e_i\), hence it contains the whole probability simplex. The robust constraint becomes

\[
\max_i f_i\le F.
\]

Since memory decreases monotonically as each \(f_i\) increases, the optimum is

\[
f_i=F\quad\forall i\text{ with }g_i>0.
\]

The learned allocation has therefore collapsed exactly to the classical Bloom-filter allocation.

This is a formal robustness–memory phase transition: when no useful assumption about future query distribution remains, the memory benefit of workload learning disappears.

## Proposition 5 — fixed-sample deployment certificate

Suppose \(m\) verified non-key audit queries are sampled independently from the current workload and \(X\) of them produce false positives. Let

\[
U_\delta(X,m)=
\operatorname{Beta}^{-1}(1-\delta;X+1,m-X)
\]

with the usual endpoint convention. This is the one-sided Clopper–Pearson upper confidence bound for the live FPR \(p\). Hence

\[
\Pr[p\le U_\delta(X,m)]\ge1-\delta.
\]

A conservative deployment rule is therefore:

- CERTIFIED if \(U_\delta(X,m)\le F\);
- otherwise REBUILD / INVESTIGATE.

This certificate does not infer safety merely from a generic drift detector. It tests the operational quantity that the approximate-membership structure promises: false-positive probability.

### Assumptions and limitation

The fixed-sample binomial certificate assumes the audited queries are representative and independent enough for the Bernoulli model. Temporally correlated or adaptively stopped monitoring requires a block-aware analysis or an anytime-valid confidence sequence. That is a planned extension, not a claim of the current experiment.

## No-result policy

Synthetic tests may validate algorithms and theorems but are not evidence that the method works on real drift. The EMBER2024 temporal experiment is evaluated under precommitted kill criteria. If it fails them, the result is reported as a failure or redesign trigger rather than re-labelled as success.
