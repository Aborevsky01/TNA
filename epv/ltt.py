"""Learn-Then-Test (LTT) baseline — a standalone comparison method, NOT part of
the TNA pipeline. Run it separately (see examples/run_ltt.py) to overlay on the
FDR-control plots.

LTT (Angelopoulos et al.) calibrates a score threshold on the training set using
Hoeffding-Bentkus p-values + a Bonferroni/Holm correction, for a grid of target
risk levels alpha, then counts how many test predictions pass that threshold.
"""
from __future__ import annotations
import numpy as np

from .stats import calculate_fdr, hb_p_value, bonferroni


def ltt_curve(train_scores, train_labels, test_scores, target_label=1,
              alpha_max=0.30, n_alpha=100, n_lambda=300, delta=0.15):
    """Return (alphas, trusted_counts): the LTT trusted-vs-estimated-FDR curve."""
    train = np.asarray(train_scores, float).ravel()
    test = np.asarray(test_scores, float).ravel()
    bin_train = (np.asarray(train_labels).ravel() == target_label).astype(float)

    fdrs, _ = calculate_fdr(train, bin_train)                 # running FDP, descending
    lambdas = (train.size * np.linspace(0, 1, n_lambda)).astype(int)[:-1]
    r_hats = fdrs[lambdas]
    desc = np.sort(train)[::-1]

    alphas = np.linspace(1e-3, alpha_max, n_alpha)
    trusted = []
    for a in alphas:
        pvals = np.array([hb_p_value(r, train.size, alpha=a) for r in r_hats])
        chosen = bonferroni(pvals, delta)                     # Holm-corrected rejections
        try:
            cutoff = desc[lambdas[chosen][-1]]
        except IndexError:
            cutoff = desc[-1]
        trusted.append(int((test > cutoff).sum()))
    return alphas, np.array(trusted)
