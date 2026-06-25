"""Multi-class / multi-label Test Null Adjustment (one-vs-rest).

Conceptually identical to the binary method, applied per class. For K classes
and a score matrix of shape (n, K):

  - for class c, the training null is column c of the instances whose label != c
    (one-vs-rest);
  - the binary TNA is fitted with target_label=c on that column and used to
    recalibrate the column-c test scores;
  - per-class Benjamini-Hochberg q-values are computed against that class null;
  - the M x K hypotheses "is instance i of class c?" are pooled (multi-label FDR),
    matching the BCSS experiment in the manuscript.

This faithfully ports notebooks/breast_cancer.ipynb. It is binary TNA repeated
one-vs-rest; see the module note in the package docs for the single/multi/paper
points that still need confirmation (pi0 estimator, per-class board, pooling vs
argmin).  NOTE: not yet validated numerically (BCSS scores not available here).
"""
from __future__ import annotations
import numpy as np

from .core import TNA
from .stats import qvalues_from_labels


class MultiTNA:
    def __init__(self, n_bins=100, pi0_method="storey", variant="minus", board=0.0):
        self.n_bins = n_bins
        self.pi0_method = pi0_method
        self.variant = variant
        self.board = board

    def fit(self, train_scores, train_labels):
        """train_scores: (n, K) per-class scores; train_labels: (n,) integer classes."""
        X = np.asarray(train_scores, float)
        y = np.asarray(train_labels).ravel()
        self.K_ = X.shape[1]
        self.models_ = []
        for c in range(self.K_):
            m = TNA(n_bins=self.n_bins, pi0_method=self.pi0_method,
                    variant=self.variant, board=self.board).fit(X[:, c], y, target_label=c)
            self.models_.append(m)
        return self

    def qvalues(self, test_scores, null_hist=None):
        """Return the per-class q-value matrix of shape (m, K).

        null_hist (for variant='plus') may be a list/array of per-class histograms.
        """
        X = np.asarray(test_scores, float)
        cols = []
        for c in range(self.K_):
            nh = null_hist[c] if (null_hist is not None) else None
            cols.append(self.models_[c].qvalues(X[:, c], null_hist=nh))
        return np.column_stack(cols)

    def predict(self, test_scores, null_hist=None):
        """Assign each instance to the class with the smallest q-value.

        Returns (predicted_class, q_of_that_class).
        """
        Q = self.qvalues(test_scores, null_hist=null_hist)
        pred = Q.argmin(axis=1)
        return pred, Q[np.arange(Q.shape[0]), pred]

    def pooled_qvalues(self, test_scores, null_hist=None):
        """Pool all m*K one-vs-rest hypotheses into one sorted multi-label q-curve."""
        return np.sort(self.qvalues(test_scores, null_hist=null_hist).ravel(order="F"))


def ground_truth_pooled(test_scores, test_labels):
    """Oracle pooled q-values for the m*K (instance, class) hypotheses."""
    X = np.asarray(test_scores, float)
    y = np.asarray(test_labels).ravel()
    K = X.shape[1]
    scores = np.concatenate([X[:, c] for c in range(K)])
    labels = np.concatenate([(y == c).astype(int) for c in range(K)])
    return qvalues_from_labels(scores, labels)
