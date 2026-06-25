"""Test Null Adjustment (TNA): reliable FDR control under distribution shift.

Quick start
-----------
    from epv import TNA
    from epv.data import load_scores

    d = load_scores("chexpert_shift")
    model = TNA().fit(d["train_scores"], d["train_labels"])
    q = model.qvalues(d["test_scores"])      # FDR-controlled q-values
    trusted = q <= 0.1                        # discoveries at 10% FDR
"""
from .core import TNA, ground_truth_qvalues
from .multi import MultiTNA, ground_truth_pooled
from .pi0 import estimate_pi0, ESTIMATORS
from .stats import (empirical_p_values, qvalues_from_pvalues,
                    qvalues_from_labels, calculate_fdr)

__all__ = [
    "TNA", "ground_truth_qvalues",
    "MultiTNA", "ground_truth_pooled",
    "estimate_pi0", "ESTIMATORS",
    "empirical_p_values", "qvalues_from_pvalues", "qvalues_from_labels", "calculate_fdr",
]
__version__ = "0.1.0"
