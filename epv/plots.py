"""Optional plotting helpers for the FDR-control diagnostics.

matplotlib is a *soft* dependency: the core ``epv`` package never imports it, and
these helpers raise a clear message if it is missing. Typical use:

    import epv.plots as ep
    curves = ep.evaluate(train_scores, train_labels, test_scores, test_labels)
    ep.fdr_control(curves, level=0.1)        # trusted-vs-q curve, all methods
    ep.qq(ep.null_pvalues(...))              # null p-value Q-Q

You can also pass your own dict of q-value arrays to ``fdr_control`` directly.
"""
from __future__ import annotations
import numpy as np

from .core import TNA, ground_truth_qvalues
from .stats import qvalues_from_pvalues, empirical_p_values

COLORS = {"GT": "black", "EPV": "red", "TNA-": "green", "TNA+": "blueviolet"}


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting requires matplotlib — install with `pip install matplotlib`") from e


def _thin(q, n=3000):
    q = np.sort(np.asarray(q, float))
    idx = np.linspace(0, q.size - 1, min(q.size, n)).astype(int)
    return q[idx], idx


def evaluate(train_scores, train_labels, test_scores, test_labels=None, *,
             target_label=1, n_bins=100, pi0_method="storey", null_hist=None):
    """Compute q-value curves for GT (if labels given), EPV, TNA- and TNA+.

    Returns a dict {label: q-values}; feed it straight into ``fdr_control``.
    """
    train_scores = np.asarray(train_scores, float).ravel()
    test_scores = np.asarray(test_scores, float).ravel()
    train_labels = np.asarray(train_labels).ravel()
    tn = train_scores[train_labels != target_label]

    curves = {}
    if test_labels is not None:
        curves["GT"] = ground_truth_qvalues(test_scores, test_labels, target_label)
    curves["EPV"], _ = qvalues_from_pvalues(tn, test_scores, pi0_method=pi0_method)
    curves["TNA-"] = TNA(n_bins=n_bins, pi0_method=pi0_method).fit(
        train_scores, train_labels, target_label).qvalues(test_scores)
    if null_hist is not None:
        curves["TNA+"] = TNA(n_bins=len(np.asarray(null_hist)), pi0_method=pi0_method,
                             variant="plus").fit(train_scores, train_labels, target_label).qvalues(
                                 test_scores, null_hist=null_hist)
    return curves


def null_pvalues(train_neg, test_neg_dict):
    """Helper for Q-Q: dict {label: null p-values} from adjusted negative scores.

    ``test_neg_dict`` maps a label to an array of (adjusted) negative-side test
    scores; p-values are computed against the sorted training null ``train_neg``.
    """
    ref = np.sort(np.asarray(train_neg, float))
    return {k: empirical_p_values(ref, v) for k, v in test_neg_dict.items()}


def fdr_control(curves, ax=None, level=None, xmax=0.3, title=None):
    """Plot number of trusted predictions vs estimated FDR (q) for each method."""
    plt = _plt()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 5.5))
    for name, q in curves.items():
        qx, ix = _thin(q)
        lw = 3 if name == "GT" else 1.8
        ls = "-" if name == "GT" else "--"
        ax.plot(qx, ix, ls, lw=lw, color=COLORS.get(name), label=name)
    if "GT" in curves:
        ncap = int(np.searchsorted(np.sort(curves["GT"]), xmax, "right"))
        ax.set_ylim(0, max(int(ncap * 1.8), 50))
    if level is not None:
        ax.axvline(level, color="grey", ls=":", lw=1)
    ax.set_xlim(0, xmax)
    ax.set(xlabel="estimated FDR (q)", ylabel="no. trusted predictions",
           title=title or "FDR control")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    return ax


def qq(pvalues, ax=None, title=None):
    """Q-Q plot of (null) empirical p-values against the uniform distribution."""
    plt = _plt()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k-.", lw=1.5)
    for name, p in pvalues.items():
        p = np.sort(np.asarray(p, float))
        pos = np.arange(1, p.size + 1) / p.size
        ax.plot(pos, p, lw=2, color=COLORS.get(name), label=name)
    ax.set(xlabel="theoretical p-value (uniform)", ylabel="empirical p-value",
           title=title or "Null p-value Q-Q")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
    return ax
