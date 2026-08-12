from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    score: float
    ks: float
    js: float
    mean_shift: float
    std_ratio: float


@dataclass(frozen=True)
class DriftReport:
    features: list[FeatureDrift]
    mmd2: float
    covariance_shift: float


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[np.isfinite(a)], b[np.isfinite(b)]


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _finite_pair(a, b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def js_divergence(a: np.ndarray, b: np.ndarray, bins: int = 24) -> float:
    a, b = _finite_pair(a, b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if abs(hi - lo) < 1e-12:
        return 0.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    p = ha.astype(float) + 1e-12
    q = hb.astype(float) + 1e-12
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _finite_pair(a, b)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
    return float(abs(np.mean(a) - np.mean(b)) / max(pooled, 1e-12))


def _std_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _finite_pair(a, b)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    sa, sb = float(np.std(a)), float(np.std(b))
    return float((sb + 1e-12) / (sa + 1e-12))


def _sq_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2 * X @ Y.T)


def rbf_mmd2(reference: np.ndarray, current: np.ndarray, *, max_samples: int = 400, seed: int = 0) -> float:
    X = np.asarray(reference, dtype=float)
    Y = np.asarray(current, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("reference and current must be 2D with matching feature counts")
    rng = np.random.default_rng(seed)
    if len(X) > max_samples: X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples: Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    Z = np.vstack([X, Y])
    d = _sq_dist(Z, Z)
    positive = d[d > 1e-12]
    bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2 * bandwidth2, 1e-12)
    Kxx = np.exp(-gamma * _sq_dist(X, X))
    Kyy = np.exp(-gamma * _sq_dist(Y, Y))
    Kxy = np.exp(-gamma * _sq_dist(X, Y))
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def drift_report(reference: np.ndarray, current: np.ndarray, feature_names: Sequence[str] | None = None, *, bins: int = 24, seed: int = 0) -> DriftReport:
    X = np.asarray(reference, dtype=float)
    Y = np.asarray(current, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("reference and current must be 2D with matching feature counts")
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
    if len(names) != X.shape[1]:
        raise ValueError("feature_names length does not match X")

    features = []
    for j, name in enumerate(names):
        ks = ks_statistic(X[:, j], Y[:, j])
        js = js_divergence(X[:, j], Y[:, j], bins=bins)
        mean = _mean_shift(X[:, j], Y[:, j])
        ratio = _std_ratio(X[:, j], Y[:, j])
        log_ratio = abs(float(np.log(max(ratio, 1e-12))))
        score = 0.45 * ks + 0.25 * min(1.0, js / 0.25) + 0.2 * min(1.0, mean / 2) + 0.1 * min(1.0, log_ratio)
        features.append(FeatureDrift(name, float(score), ks, js, mean, ratio))
    features.sort(key=lambda r: r.score, reverse=True)

    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)
    denom = np.linalg.norm(cov_x, ord="fro") + 1e-12
    cov_shift = float(np.linalg.norm(cov_y - cov_x, ord="fro") / denom)
    return DriftReport(features, rbf_mmd2(X, Y, seed=seed), cov_shift)
