"""Test Null Adjustment (TNA).

A small, sklearn-style estimator. Fit it on training scores + labels, then
``transform`` recalibrates test scores so that the test null aligns with the
training null, and ``qvalues`` returns Benjamini-Hochberg q-values usable for
FDR control under distribution shift.

    model = TNA().fit(train_scores, train_labels)
    q     = model.qvalues(test_scores)      # FDR-controlled q-values
    keep = q <= 0.1                         # trusted predictions at FDR 10%

Variants:
    "minus" (default) - assumes train/test NULL distributions have a similar
                        shape; the null histogram is transferred from training.
    "plus"            - assumes train/test NON-NULL distributions are similar;
                        requires a reconstructed null histogram (``null_hist``).
"""
from __future__ import annotations
import numpy as np

from .stats import empirical_p_values, qvalues_from_pvalues, qvalues_from_labels
from .pi0 import estimate_pi0


class TNA:
    def __init__(self, n_bins=100, pi0_method="storey", variant="minus",
                 board=0.0, drop_top_bins=5):
        assert variant in ("minus", "plus")
        self.n_bins = n_bins
        self.pi0_method = pi0_method
        self.variant = variant
        self.board = board
        self.drop_top_bins = drop_top_bins

    # --- fit ------------------------------------------------------------- #
    def fit(self, train_scores, train_labels, target_label=1):
        """Store the training null / non-null score distributions."""
        x = np.asarray(train_scores, float).ravel()
        is_pos = np.asarray(train_labels).ravel() == target_label
        self.train_neg_ = x[~is_pos]
        self.train_pos_ = x[is_pos]
        self.train_neg_sorted_ = np.sort(self.train_neg_)
        self.pi0_X_ = self.train_neg_.size / x.size
        self.edges_ = np.linspace(self.train_neg_.min(), self.train_neg_.max(), self.n_bins + 1)
        return self

    # --- transform: recalibrate test scores ------------------------------ #
    def transform(self, test_scores, null_hist=None):
        """Return test scores with the null part realigned to the training null.

        Output is aligned to the input order. Only the (reconstructed) null-side
        scores are adjusted; the non-null side is left untouched.
        """
        t = np.asarray(test_scores, float).ravel()
        edges = self.edges_
        L = self.n_bins + 2
        bidx = np.digitize(t, edges)
        bc_test = np.bincount(bidx, minlength=L).astype(float)
        pi0_T = (t < self.board).sum() / t.size

        if self.variant == "minus":
            relate = self._null_counts_minus(bc_test, pi0_T, L)
        else:
            relate = self._null_counts_plus(bc_test, pi0_T, L, null_hist)

        # flag which test points form the reconstructed null (order-preserving)
        hi = self.n_bins - self.drop_top_bins
        is_null = np.zeros(t.size, bool)
        for i in range(1, min(hi, L)):
            idx = np.nonzero(bidx == i)[0]
            k = int(np.clip(relate[i], 0, idx.size))
            is_null[idx[:k]] = True
        tail = (bidx <= 0) | (bidx >= hi)
        is_null[tail & (t < self.board)] = True

        null_scores = t[is_null]
        self.pi0_T_ = pi0_T
        if null_scores.size < 2:                       # nothing to align
            return t.copy()
        m, s = null_scores.mean(), null_scores.std()
        adj = t.copy()
        adj[is_null] = (t[is_null] - m) / s * self.train_neg_.std() + self.train_neg_.mean()
        return adj

    # --- p-values / q-values --------------------------------------------- #
    def pvalues(self, test_scores, null_hist=None):
        adj = self.transform(test_scores, null_hist=null_hist)
        return empirical_p_values(self.train_neg_sorted_, adj)

    def qvalues(self, test_scores, pi0=None, null_hist=None):
        """Benjamini-Hochberg q-values of the adjusted test scores."""
        adj = self.transform(test_scores, null_hist=null_hist)
        q, self.pi0_ = qvalues_from_pvalues(
            self.train_neg_sorted_, adj, pi0=pi0, pi0_method=self.pi0_method)
        return q

    def fit_transform(self, train_scores, train_labels, test_scores, target_label=1):
        return self.fit(train_scores, train_labels, target_label).transform(test_scores)

    # --- internals ------------------------------------------------------- #
    def _null_counts_minus(self, bc_test, pi0_T, L):
        edges = self.edges_
        bcn = np.bincount(np.digitize(self.train_neg_, edges), minlength=L).astype(float)
        bcp = np.bincount(np.digitize(self.train_pos_, edges), minlength=L).astype(float)
        bcn = bcn.copy()
        bcn[1:-1] = bcn[1:-1] / self.pi0_X_ * pi0_T        # proportion correction
        ratio = np.nan_to_num(bcn / (bcn + bcp), nan=0.0)  # empty bin -> 0
        return (ratio * bc_test).round().astype(int)

    def _null_counts_plus(self, bc_test, pi0_T, L, null_hist):
        if null_hist is None:
            raise ValueError("variant='plus' needs a reconstructed null histogram "
                             "(null_hist=...); the analytic form is numerically unstable.")
        nh = np.asarray(null_hist, float)
        nh = np.pad(nh, (0, max(0, L - nh.size)))[:L]
        n_test_neg = int(round(pi0_T * bc_test.sum()))   # estimated number of test nulls
        return (nh * n_test_neg).round().astype(int)


def ground_truth_qvalues(test_scores, test_labels, target_label=1):
    """Oracle q-values from true labels (for evaluation only)."""
    y = (np.asarray(test_labels).ravel() == target_label).astype(int)
    return qvalues_from_labels(np.asarray(test_scores, float).ravel(), y)
