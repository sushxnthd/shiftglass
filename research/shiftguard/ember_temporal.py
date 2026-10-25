#!/usr/bin/env python3
import argparse, csv, json, math, zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LN2 = math.log(2.0)
BF_DEN = LN2 * LN2


def bf_bits(n_keys: int, fpr: float) -> float:
    fpr = float(np.clip(fpr, 1e-12, 1 - 1e-12))
    return -n_keys * math.log(fpr) / BF_DEN


def normalize(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("distribution has zero mass")
    return v / s


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def worstcase_tv(center, values, rho, return_q=False):
    """Exact max_q q·values subject to TV(q, center) <= rho on a finite support."""
    p = normalize(center).copy()
    v = np.asarray(values, dtype=float)
    q = p.copy()
    budget = float(np.clip(rho, 0.0, 1.0))
    donors = list(np.argsort(v))
    recvs = list(np.argsort(-v))
    di = ri = 0
    while budget > 1e-14 and di < len(donors) and ri < len(recvs):
        d, r = donors[di], recvs[ri]
        if v[r] <= v[d] + 1e-15:
            break
        if q[d] <= 1e-15:
            di += 1
            continue
        if q[r] >= 1 - 1e-15:
            ri += 1
            continue
        move = min(budget, q[d], 1.0 - q[r])
        q[d] -= move
        q[r] += move
        budget -= move
        if q[d] <= 1e-15:
            di += 1
        if q[r] >= 1 - 1e-15:
            ri += 1
    val = float(np.dot(q, v))
    return (val, q) if return_q else val


def weighted_bf_bits(n_keys, g, f):
    g = normalize(g)
    f = np.clip(np.asarray(f, dtype=float), 1e-12, 1.0)
    return float(n_keys * np.sum(g * (-np.log(f))) / BF_DEN)


def standard_allocation(g, h, target):
    """Optimal f_i for fixed regions under h·f <= target."""
    g, h = normalize(g), normalize(h)
    lo, hi = 0.0, 1e12
    for _ in range(120):
        lam = (lo + hi) / 2.0
        f = np.ones_like(g)
        nz = h > 0
        f[nz] = np.minimum(1.0, lam * g[nz] / h[nz])
        if float(np.dot(h, f)) > target:
            hi = lam
        else:
            lo = lam
    lam = lo
    f = np.ones_like(g)
    nz = h > 0
    f[nz] = np.minimum(1.0, lam * g[nz] / h[nz])
    return np.clip(f, 1e-9, 1.0)


def robust_allocation(g, h, target, rho):
    g, h = normalize(g), normalize(h)
    if rho <= 1e-12:
        return standard_allocation(g, h, target), True, "standard"
    k = len(g)
    flat = np.full(k, min(target * 0.999, 0.999), dtype=float)
    std = standard_allocation(g, h, target)
    starts = [flat, 0.75 * flat + 0.25 * std, 0.5 * flat + 0.5 * std]

    def obj(f):
        return np.sum(g * (-np.log(np.clip(f, 1e-12, 1.0))))

    def con(f):
        return target - worstcase_tv(h, f, rho)

    best = None
    for x0 in starts:
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=[(1e-8, 1.0)] * k,
            constraints=[{"type": "ineq", "fun": con}],
            options={"maxiter": 3000, "ftol": 1e-11, "disp": False},
        )
        if res.x is None:
            continue
        feasible = worstcase_tv(h, res.x, rho) <= target * (1 + 1e-7)
        if feasible and (best is None or obj(res.x) < obj(best.x)):
            best = res
    if best is None:
        return np.full(k, target), False, "fallback_flat"
    return np.clip(best.x, 1e-8, 1.0), bool(best.success), best.message


def cp_upper(x, n, delta=0.05):
    if n <= 0 or x >= n:
        return 1.0
    return float(beta.ppf(1 - delta, x + 1, n - x))


def score_regions(scores, edges):
    idx = np.searchsorted(edges[1:-1], scores, side="right")
    return np.clip(idx, 0, len(edges) - 2)


def hist_dist(scores, edges):
    k = len(edges) - 1
    idx = score_regions(np.asarray(scores), edges)
    return np.bincount(idx, minlength=k).astype(float) / max(1, len(idx))


def file_time_key(path: Path):
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j >= 40:
                break
            try:
                r = json.loads(line)
                t = r.get("first_submission_date")
                if t is not None:
                    if isinstance(t, (int, float)):
                        vals.append(float(t))
                    else:
                        import datetime as dt
                        s = str(t).replace("Z", "+00:00")
                        vals.append(dt.datetime.fromisoformat(s).timestamp())
            except Exception:
                pass
    return min(vals) if vals else float("inf")


def load_vectors(files, extractor):
    X, y = [], []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                label = r.get("label")
                if label not in (0, 1):
                    continue
                X.append(np.asarray(extractor.process_raw_features(r), dtype=np.float32))
                y.append(int(label))
    if not X:
        raise RuntimeError("No labeled rows found")
    return np.vstack(X), np.asarray(y, dtype=np.int8)


def oracle_params_bytes(selector, scaler, clf, original_dim):
    k = int(selector.get_support().sum())
    index_bytes = k * (2 if original_dim < 65536 else 4)
    float_bytes = k * 4 * 3 + 4
    return int(index_bytes + float_bytes)


def predict_scores(X, selector, scaler, clf):
    Z = selector.transform(X)
    Z = scaler.transform(Z).astype(np.float32, copy=False)
    return clf.predict_proba(Z)[:, 1]


def download_and_extract(workdir: Path):
    from huggingface_hub import hf_hub_download

    raw = workdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        out = raw / split
        out.mkdir(parents=True, exist_ok=True)
        if list(out.rglob("*.jsonl")):
            continue
        fn = f"PDF_{split}.zip"
        p = hf_hub_download(
            repo_id="joyce8/EMBER2024", filename=fn, repo_type="dataset"
        )
        with zipfile.ZipFile(p) as z:
            z.extractall(out)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=".shiftguard-work")
    ap.add_argument("--outdir", default="research/shiftguard/results")
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--regions", type=int, default=8)
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    np.random.seed(7)
    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = download_and_extract(workdir)
    from thrember.features import PEFeatureExtractor

    extractor = PEFeatureExtractor()

    train_files = sorted((raw / "train").rglob("*.jsonl"), key=file_time_key)
    test_files = sorted((raw / "test").rglob("*.jsonl"), key=file_time_key)
    if len(train_files) < 52 or len(test_files) < 12:
        raise RuntimeError(
            f"Expected >=52 train and >=12 test weekly JSONL files; got "
            f"{len(train_files)} and {len(test_files)}"
        )

    train_files = train_files[:52]
    test_files = test_files[:12]
    fit_files = train_files[:40]
    ref_files = train_files[40:46]
    radius_files = train_files[46:52]

    X_fit, y_fit = load_vectors(fit_files, extractor)
    X_ref, y_ref = load_vectors(ref_files, extractor)
    X_radius, y_radius = load_vectors(radius_files, extractor)

    kfeat = min(args.features, X_fit.shape[1])
    selector = SelectKBest(f_classif, k=kfeat)
    Xs = selector.fit_transform(X_fit, y_fit)
    scaler = StandardScaler().fit(Xs)
    Xz = scaler.transform(Xs).astype(np.float32, copy=False)
    clf = LogisticRegression(
        max_iter=1200, solver="liblinear", C=1.0, random_state=7
    )
    clf.fit(Xz, y_fit)

    fit_scores = predict_scores(X_fit, selector, scaler, clf)
    ref_scores = predict_scores(X_ref, selector, scaler, clf)
    radius_scores = predict_scores(X_radius, selector, scaler, clf)
    fit_auc = roc_auc_score(y_fit, fit_scores)
    ref_auc = roc_auc_score(y_ref, ref_scores)
    radius_auc = roc_auc_score(y_radius, radius_scores)

    key_scores = np.concatenate(
        [
            fit_scores[y_fit == 1],
            ref_scores[y_ref == 1],
            radius_scores[y_radius == 1],
        ]
    )
    ref_neg = ref_scores[y_ref == 0]
    n_keys = int(key_scores.size)
    oracle_bytes = oracle_params_bytes(selector, scaler, clf, X_fit.shape[1])
    oracle_bits = oracle_bytes * 8

    q = np.linspace(0, 1, args.regions + 1)
    edges = np.quantile(key_scores, q)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    edges[0], edges[-1] = -np.inf, np.inf

    g = hist_dist(key_scores, edges)
    h = hist_dist(ref_neg, edges)

    # Select the deployed robustness radius using only pre-test history:
    # weeks 40-45 define the reference workload; weeks 46-51 measure
    # recent temporal movement. The future 12 test weeks never tune rho.
    radius_week_tvs = []
    for fp in radius_files:
        Xr, yr = load_vectors([fp], extractor)
        sr = predict_scores(Xr, selector, scaler, clf)
        qr = hist_dist(sr[yr == 0], edges)
        radius_week_tvs.append(tv(qr, h))
    selected_rho = float(max(radius_week_tvs)) if radius_week_tvs else 0.0

    f_std = standard_allocation(g, h, args.target)
    std_filter_bits = weighted_bf_bits(n_keys, g, f_std)
    classical_bits = bf_bits(n_keys, args.target)

    weekly = []
    for wi, fp in enumerate(test_files):
        Xw, yw = load_vectors([fp], extractor)
        sw = predict_scores(Xw, selector, scaler, clf)
        neg = sw[yw == 0]
        qw = hist_dist(neg, edges)
        weekly.append(
            {
                "week": wi,
                "file": fp.name,
                "n": int(len(yw)),
                "n_benign": int((yw == 0).sum()),
                "n_malicious": int((yw == 1).sum()),
                "auc": float(roc_auc_score(yw, sw))
                if len(np.unique(yw)) == 2
                else None,
                "tv_from_cal": tv(qw, h),
                "q": qw,
                "std_expected_fpr": float(np.dot(qw, f_std)),
                "neg_region_idx": score_regions(neg, edges),
            }
        )

    rho_grid = sorted(set(
        [0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
        + [selected_rho]
    ))
    rows = []
    cert_rows = []
    rng = np.random.default_rng(7)

    for rho in rho_grid:
        f, ok, msg = robust_allocation(g, h, args.target, rho)
        filt_bits = weighted_bf_bits(n_keys, g, f)
        total_bits = filt_bits + oracle_bits
        future_fprs = [float(np.dot(w["q"], f)) for w in weekly]
        future_tvs = [w["tv_from_cal"] for w in weekly]
        covered = [w for w in weekly if w["tv_from_cal"] <= rho + 1e-12]

        rows.append(
            {
                "rho": rho,
                "optimizer_ok": bool(ok),
                "optimizer_message": str(msg),
                "filter_bits": filt_bits,
                "total_bits_with_oracle": total_bits,
                "classical_bits": classical_bits,
                "filter_saving_pct": 100 * (1 - filt_bits / classical_bits),
                "end_to_end_saving_pct": 100
                * (1 - total_bits / classical_bits),
                "designed_worstcase_fpr": worstcase_tv(h, f, rho),
                "covered_weeks": len(covered),
                "coverage_pct": 100 * len(covered) / len(weekly),
                "max_future_expected_fpr": max(future_fprs),
                "mean_future_expected_fpr": float(np.mean(future_fprs)),
                "future_weeks_over_target": int(
                    sum(x > args.target * (1 + 1e-9) for x in future_fprs)
                ),
                "max_future_tv": max(future_tvs),
                "allocation": f.tolist(),
            }
        )

        # A second, direct deploy-time certificate: sample verified non-key queries,
        # observe actual false positives, and use a one-sided exact binomial bound.
        for m in (200, 300, 500, 1000, 2000):
            accepts = false_accepts = trials = 0
            for w in weekly:
                ridx = w["neg_region_idx"]
                if len(ridx) == 0:
                    continue
                true_fpr = float(np.mean(f[ridx]))
                for _ in range(100):
                    samp = rng.choice(ridx, size=m, replace=True)
                    probs = f[samp]
                    x = int((rng.random(m) < probs).sum())
                    upper = cp_upper(x, m, args.delta)
                    passed = upper <= args.target
                    accepts += int(passed)
                    false_accepts += int(passed and true_fpr > args.target)
                    trials += 1
            cert_rows.append(
                {
                    "rho": rho,
                    "audit_m": m,
                    "trials": trials,
                    "accept_rate": accepts / trials if trials else 0.0,
                    "false_accept_rate": false_accepts / trials
                    if trials
                    else 0.0,
                }
            )

    # The decisive verdict is based on exactly one radius chosen from
    # pre-test history, never on whichever rho looks best on future data.
    selected_row = min(rows, key=lambda r: abs(r["rho"] - selected_rho))
    selected_cert = next(
        c for c in cert_rows
        if abs(c["rho"] - selected_row["rho"]) < 1e-12
        and c["audit_m"] == 1000
    )
    passes = (
        selected_row["coverage_pct"] >= 80.0
        and selected_row["end_to_end_saving_pct"] >= 15.0
        and selected_row["designed_worstcase_fpr"] <= args.target * 1.0001
        and selected_cert["accept_rate"] >= 0.50
        and selected_cert["false_accept_rate"] <= 0.05
    )
    decision = "PROMOTE" if passes else "KILL_OR_REDESIGN"

    with (outdir / "rho_sweep.csv").open("w", newline="") as fcsv:
        cols = [k for k in rows[0].keys() if k != "allocation"]
        wri = csv.DictWriter(fcsv, fieldnames=cols)
        wri.writeheader()
        wri.writerows([{k: r[k] for k in cols} for r in rows])

    with (outdir / "audit_certificate.csv").open("w", newline="") as fcsv:
        wri = csv.DictWriter(fcsv, fieldnames=list(cert_rows[0].keys()))
        wri.writeheader()
        wri.writerows(cert_rows)

    with (outdir / "weekly_temporal.csv").open("w", newline="") as fcsv:
        cols = [
            "week",
            "file",
            "n",
            "n_benign",
            "n_malicious",
            "auc",
            "tv_from_cal",
            "std_expected_fpr",
        ]
        wri = csv.DictWriter(fcsv, fieldnames=cols)
        wri.writeheader()
        wri.writerows([{k: w[k] for k in cols} for w in weekly])

    summary = {
        "dataset": "EMBER2024 PDF official temporal subset",
        "split": (
            "weeks 0-39 tiny-oracle fit; weeks 40-45 reference workload; "
            "weeks 46-51 choose robustness radius; official 12 future test "
            "weeks untouched until evaluation"
        ),
        "target_fpr": args.target,
        "regions": args.regions,
        "selected_features": kfeat,
        "fit_rows": int(len(y_fit)),
        "reference_rows": int(len(y_ref)),
        "radius_selection_rows": int(len(y_radius)),
        "n_keys": n_keys,
        "reference_nonkeys": int(len(ref_neg)),
        "fit_auc": float(fit_auc),
        "reference_auc": float(ref_auc),
        "radius_selection_auc": float(radius_auc),
        "historical_radius_week_tvs": radius_week_tvs,
        "selected_rho_from_history": selected_rho,
        "oracle_bytes": oracle_bytes,
        "classical_bits": classical_bits,
        "standard_filter_bits": std_filter_bits,
        "standard_total_bits": std_filter_bits + oracle_bits,
        "standard_end_to_end_saving_pct": 100
        * (1 - (std_filter_bits + oracle_bits) / classical_bits),
        "standard_max_future_expected_fpr": max(
            w["std_expected_fpr"] for w in weekly
        ),
        "future_max_tv": max(w["tv_from_cal"] for w in weekly),
        "future_median_tv": float(
            np.median([w["tv_from_cal"] for w in weekly])
        ),
        "kill_criteria": {
            "future_week_coverage_min_pct": 80,
            "end_to_end_memory_saving_min_pct": 15,
            "audit_queries": 1000,
            "audit_accept_rate_min": 0.50,
            "empirical_false_certification_max": 0.05,
            "confidence": 1 - args.delta,
        },
        "decision": decision,
        "selected_operating_point": selected_row,
        "selected_audit_1000": selected_cert,
        "rho_sweep": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
