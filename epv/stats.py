"""Statistical helpers for empirical-p-value FDR control (vectorised numpy).

These are independent of the TNA adjustment itself: empirical p-values against a
reference null, the Benjamini-Hochberg q-values, ground-truth q-values from
labels, and the Hoeffding-Bentkus p-value used by the LTT baseline.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests

from .pi0 import estimate_pi0


def empirical_p_values(reference_null, query):
    """p = #{x in reference_null : x > score} / |reference_null|.

    ``reference_null`` must be sorted ascending. Returned in ``query`` order.
    """
    reference_null = np.asarray(reference_null, float)
    query = np.asarray(query, float)
    n = reference_null.size
    idx = np.searchsorted(reference_null, query, side="right")  # == #{x <= score}
    return (n - idx) / n


def calculate_fdr(scores, labels):
    """Running FDP along the descending score ranking. Returns (fdp, order)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="stable")
    sl = labels[order]
    pos = np.cumsum(sl == 1)
    neg = np.cumsum(sl == 0)
    return neg / np.maximum(neg + pos, 1), order


def qvalues_from_labels(scores, labels):
    """Ground-truth q-values from true labels (monotone-min of the FDP)."""
    q, _ = calculate_fdr(scores, labels)
    q = np.sort(q)
    for i in range(len(q) - 1, 0, -1):
        q[i - 1] = min(q[i], q[i - 1])
    return q


def qvalues_from_pvalues(reference_null, query, pi0=None, pi0_method="storey", **pi0_kw):
    """Storey/BH q-values for ``query`` scores against ``reference_null``.

    ``pi0`` : fixed value, or None to estimate it via ``pi0_method``.
    Returns (q_values in query order, pi0_used).
    """
    p = empirical_p_values(np.sort(reference_null), query)
    pi_0 = float(pi0) if pi0 is not None else estimate_pi0(p, method=pi0_method, **pi0_kw)
    q = p * len(p) * pi_0
    order = np.argsort(q)
    q = np.sort(q) / np.arange(1, len(p) + 1)
    for i in range(len(p) - 1, 0, -1):
        q[i - 1] = min(q[i - 1], q[i])
    return q[np.argsort(order)], pi_0


# --- LTT baseline (optional) ------------------------------------------------ #
def hb_p_value(r_hat, n, alpha=0.1):
    """Hoeffding-Bentkus p-value."""
    bentkus = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)

    def h1(y, mu):
        with np.errstate(divide="ignore"):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))

    hoeffding = np.exp(-n * h1(min(r_hat, alpha), alpha))
    return min(bentkus, hoeffding)


def bonferroni(p_values, delta):
    rej, _, _, _ = multipletests(p_values, delta, method="holm", is_sorted=False)
    return np.nonzero(rej)[0]
