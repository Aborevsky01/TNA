"""Minimal end-to-end example: FDR control with TNA under distribution shift.

    python examples/run_example.py
    python examples/run_example.py --dataset pcam_shift --alpha 0.1 --pi0 storey
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from epv import TNA, ground_truth_qvalues
from epv.data import load_scores, available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chexpert_shift",
                    help=f"one of: {available()}")
    ap.add_argument("--alpha", type=float, default=0.1, help="target FDR level")
    ap.add_argument("--pi0", default="storey",
                    choices=["storey", "pounds", "hist", "nettleton", "jiang"],
                    help="pi0 estimator (default: storey)")
    args = ap.parse_args()

    d = load_scores(args.dataset)
    print(f"dataset={args.dataset}  train={d['train_scores'].size}  test={d['test_scores'].size}")

    # 1) standard empirical p-values (no adjustment) vs 2) TNA-adjusted
    from epv.stats import qvalues_from_pvalues
    train_neg = d["train_scores"][d["train_labels"] == 0]
    q_epv, pi0_epv = qvalues_from_pvalues(train_neg, d["test_scores"], pi0_method=args.pi0)

    model = TNA(pi0_method=args.pi0).fit(d["train_scores"], d["train_labels"])
    q_tna = model.qvalues(d["test_scores"])

    q_true = ground_truth_qvalues(d["test_scores"], d["test_labels"])

    def trusted(q):
        return int((q <= args.alpha).sum())

    print(f"\npi0 (estimated, '{args.pi0}'):  EPV={pi0_epv:.3f}  TNA={model.pi0_:.3f}")
    print(f"trusted predictions at FDR <= {args.alpha}:")
    print(f"  ground truth : {trusted(q_true)}")
    print(f"  standard EPV : {trusted(q_epv)}")
    print(f"  TNA-         : {trusted(q_tna)}")


if __name__ == "__main__":
    main()
