"""Estimators of the null proportion pi_0 (the share of true nulls).

A pluggable collection so the TNA class can be extended with alternative
estimators. ``estimate_pi0(pvalues, method=...)`` dispatches by name; the
default is Storey's estimator. All estimators take a 1-D array of p-values and
return a scalar in [0, 1].

References
----------
storey   : Storey & Tibshirani (2003), PNAS.
pounds   : Pounds & Cheng (2003), Bioinformatics  (pi0 = 2 * mean(p)).
bh       : Benjamini & Hochberg (2000) adaptive lowest-slope estimator.
hist     : right-tail histogram density (Nettleton 2006 / Murray 2021 style).

The R packages ``qvalue``, ``cp4p`` and ``FDRestimation`` implement these and
several more (Jiang, Slim, Nettleton's iterative histogram, Langaas); they can
be added here behind the same interface.
"""
from __future__ import annotations
import numpy as np


def storey(p, lam=0.8):
    """Storey & Tibshirani (2003). lam is the tuning threshold lambda_S."""
    p = np.asarray(p, float)
    n = p.size
    return float(np.clip((1 + (p >= lam).sum()) / (n * (1 - lam)), 0.0, 1.0))


def pounds(p):
    """Pounds & Cheng (2003): pi0 = min(1, 2 * mean(p))."""
    p = np.asarray(p, float)
    return float(min(1.0, 2.0 * p.mean()))


def hist(p, nbins=20):
    """Right-tail histogram density: density of p-values in the last bin."""
    p = np.asarray(p, float)
    counts, _ = np.histogram(p, bins=nbins, range=(0.0, 1.0))
    return float(np.clip(counts[-1] * nbins / p.size, 0.0, 1.0))


def nettleton(p, nbins=20):
    """Nettleton / Mosig iterative histogram estimator.

    Drops the signal-contaminated left bins (those above the running average)
    until the histogram is flat, then estimates the null level from the rest.
    """
    p = np.asarray(p, float)
    m = p.size
    h, _ = np.histogram(p, bins=nbins, range=(0.0, 1.0))
    N, b, i = m, nbins, 0
    while i < b - 1:
        avg = N / b
        if h[i] > avg:
            N -= h[i]; b -= 1; i += 1
        else:
            break
    return float(min((N / b) * nbins / m, 1.0))


def jiang(p, nbins=20):
    """Jiang & Doerge (2008) style average of tail-based estimates.

    Averages the Storey-type tail estimator over thresholds in the upper half
    of [0, 1]. Close to, but not bit-identical with, the cp4p variant.
    """
    p = np.asarray(p, float)
    m = p.size
    est = [(p > i / nbins).sum() / (m * (1 - i / nbins)) for i in range(nbins // 2, nbins)]
    return float(min(np.mean(est), 1.0))


ESTIMATORS = {"storey": storey, "pounds": pounds, "hist": hist,
              "nettleton": nettleton, "jiang": jiang}
# Validated against the manuscript's pi0 table (e.g. CheXpert): pounds -> 0.809,
# hist (Last hist) -> 0.763, nettleton -> 0.786 (table 0.785), jiang -> 0.776
# (table 0.779); nettleton matches the table to ~1e-3 on all datasets.
# Two further table columns are NOT reproduced here: "slim" (Wang 2011 sliding
# linear model) and "meinshausen" (estimates 1-pi0). They live in the R packages
# cp4p / FDRestimation; port + validate against those before adding them here.


def estimate_pi0(pvalues, method="storey", **kwargs):
    """Estimate pi_0 with the named method (default: 'storey')."""
    if method not in ESTIMATORS:
        raise ValueError(f"unknown pi0 method {method!r}; choose from {list(ESTIMATORS)}")
    return ESTIMATORS[method](pvalues, **kwargs)
