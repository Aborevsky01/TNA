"""Standalone LTT baseline demo (not part of the TNA pipeline).

    python examples/run_ltt.py --dataset chexpert

Prints LTT trusted predictions at a few FDR levels and, if matplotlib is present,
overlays the LTT curve on the ground-truth curve.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epv import ground_truth_qvalues
from epv.ltt import ltt_curve
from epv.data import load_scores, available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chexpert", help=f"one of {available()}")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    d = load_scores(args.dataset)
    alphas, trusted = ltt_curve(d["train_scores"], d["train_labels"], d["test_scores"])
    qt = np.sort(ground_truth_qvalues(d["test_scores"], d["test_labels"]))

    def at(level, alphas, vals):
        i = int(np.searchsorted(alphas, level, "right")) - 1
        return int(vals[max(i, 0)])

    print(f"{args.dataset}: LTT trusted at FDR<= {args.alpha}: {at(args.alpha, alphas, trusted)} "
          f"| ground truth: {int((qt <= args.alpha).sum())}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(qt, np.arange(len(qt)), "k-", lw=2.5, label="ground truth")
        ax.plot(alphas, trusted, ":", color="tab:blue", lw=2, label="LTT")
        ax.axvline(args.alpha, color="grey", ls=":", lw=1)
        ax.set(xlim=(0, 0.3), xlabel="Estimated FDR", ylabel="trusted predictions",
               title=f"LTT vs ground truth — {args.dataset}")
        ax.legend(); ax.grid(alpha=.3)
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "img", f"ltt_{args.dataset}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print("saved", out)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
