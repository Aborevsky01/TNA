# TNA — Test Null Adjustment

**Toward reliable false discovery rate control in classification problems under distribution shift.**

Reference implementation for

> A. Borevskiy and A. Kertesz-Farkas. *Toward reliable false discovery rate control in
> classification problems under distribution shift.*

Domain shifts or batch effects can silently degrade a deployed classifier. A shift changes
the shape of the prediction-score distribution of the test samples relative to the training
samples, so the decision boundary would need re-calibration to keep the expected error
guarantees. This package implements a simple, robust heuristic that recovers **valid
p-values** (uniformly distributed under the null) for raw test prediction scores under shift,
and uses them for **false discovery rate (FDR)** control with the Benjamini–Hochberg (BH)
procedure. The method is model-agnostic, fully data-driven, and operates in the
1-dimensional space of the prediction scores.

---

## Method: EPV and TNA

**EPV (empirical p-values).** Given the prediction scores of the **negative training**
samples as an empirical null, the empirical p-value of a test score *t* is the fraction of
training-null scores that exceed it. With these p-values, BH controls the FDR — **as long as
the training and test null distributions match**. Under distribution shift they no longer
match, the p-values stop being uniform, and FDR control silently breaks.

**TNA (Test Null Adjustment).** TNA recalibrates the test scores so that the **test null**
distribution is realigned to the **training null**, restoring valid empirical p-values. It
reconstructs the test null from the score histograms, then standardises the null-side test
scores to the training-null moments. Two variants differ in their assumption:

- **TNA−** — assumes the *null* distributions of train and test have a similar shape (only
  the class proportion / location may differ).
- **TNA+** — assumes the *non-null* distributions are similar, and reconstructs the null by
  subtracting the estimated non-null component (with tail stabilisation).

After adjustment, the proportion of nulls π₀ is estimated and BH is applied. TNA works in
binary and multi-label (one-vs-rest) settings and needs no domain-specific knowledge.

---

## Install

```bash
pip install -e .            # installs the `epv` package + numpy/scipy/statsmodels
```

(or `pip install -r requirements.txt` and import `epv` from the repo root.)

---

## Quick start

```python
from epv import TNA
from epv.data import load_scores

d = load_scores("chexpert_shift")                 # cached scores for one dataset

model = TNA().fit(d["train_scores"], d["train_labels"])
q = model.qvalues(d["test_scores"])               # FDR-controlled q-values
trusted = d["test_scores"][q <= 0.10]             # discoveries at 10% FDR
```

`fit` / `transform` style:

```python
adjusted = model.transform(test_scores)            # recalibrated scores
pvals    = model.pvalues(test_scores)              # valid empirical p-values
```

CLI:

```bash
python examples/run_example.py --dataset chexpert_shift
```

---

## The π₀ estimator

The null proportion π₀ is estimated from the data; the estimator is pluggable. The
recommended setting is the **average of Storey (λ=0.8) and the step-2 proportion
N/(N+P)** — best FDR-control accuracy in our benchmarks:

```python
TNA(pi0_method="storey_step2")   # recommended: mean(Storey, step-2)
TNA(pi0_method="storey")         # Storey & Tibshirani (2003)
TNA(pi0_method="pounds")         # Pounds & Cheng (2003): 2*mean(p)
TNA(pi0_method="nettleton")      # Nettleton / Mosig iterative histogram
TNA(pi0_method="jiang")          # Jiang & Doerge (2008) tail average
TNA(pi0_method="hist")           # right-tail histogram density
```

```python
from epv import estimate_pi0
estimate_pi0(pvalues, method="pounds")
```

`pounds`, `hist`, `nettleton` reproduce the corresponding columns of the manuscript's π₀
table closely (e.g. CheXpert: `pounds`→0.809, `hist`→0.763, `nettleton`→0.786 vs 0.785).

---

## Plotting (optional)

`epv.plots` provides the FDR-control diagnostics. matplotlib is a **soft dependency** —
importing `epv` never pulls it in.

```python
import epv.plots as ep
from epv.data import load_scores

d = load_scores("tissuenet_shift")
curves = ep.evaluate(d["train_scores"], d["train_labels"],
                     d["test_scores"], d["test_labels"], null_hist=d["null_hist"])
ep.fdr_control(curves, level=0.1)     # trusted predictions vs q, all methods
```

---

## Data

`epv.data.load_scores(name)` loads **cached discriminative scores** — the raw outputs of a
trained classifier on each dataset, saved as `.npy` so the networks do not have to be
re-run. They are *not* synthetic: each `*_x_*` / `*_t_*` file holds the train / test scores,
and the `_0` / `_1` suffix is the negative / positive label split (labels are used only for
ground-truth evaluation, never by the method). The `score_*` functions in `epv/data.py`
regenerate these scores from the raw images with the corresponding pretrained model.

```python
from epv.data import load_scores, available
available()                       # datasets whose score files are present
d = load_scores("tissuenet_shift")
```

| dataset name | source data | model that produced the scores |
|---|---|---|
| `chexpert[_shift]` | CheXpert chest X-ray (target "Effusion") | DenseNet121 (torchxrayvision) |
| `pcam[_shift]` | PatchCamelyon histopathology | ResNet34 (TIA toolbox) |
| `tissuenet[_shift]` | TissueNet cell segmentation | Cellpose |

The `_shift` variants use a class-rebalanced test set (class-distribution shift). The score
files are large and **not tracked in git**; host them with the release / on Zenodo and place
them under `scores/`.

---

## Repository structure

```
epv/                  the TNA package (pure numpy/scipy; matplotlib optional)
├── __init__.py       public API: TNA, MultiTNA, estimate_pi0, ground_truth_qvalues, …
├── core.py           TNA estimator — fit / transform / pvalues / qvalues.
│                       Reconstructs the test null (TNA-/TNA+), shifts the neg/pos
│                       boundary to the KDE crossing, and applies the step-5 affine
│                       map to the null side only.
├── stats.py          empirical p-values, Benjamini-Hochberg q-values, ground-truth
│                       q-values from labels, and the Hoeffding-Bentkus p-value (LTT).
├── pi0.py            pluggable π₀ estimators (storey, storey_step2, pounds,
│                       nettleton, jiang, hist) and estimate_pi0().
├── multi.py          MultiTNA — one-vs-rest TNA for multi-label problems.
├── data.py           load_scores(name) for the cached score files + score_* scaffolds.
├── plots.py          optional FDR-control / Q-Q diagnostics (needs matplotlib).
└── ltt.py            Learn-Then-Test baseline (standalone; overlaid for comparison).
examples/
├── run_example.py    minimal end-to-end example (EPV vs TNA vs ground truth).
└── run_ltt.py        run the LTT baseline and print its trusted-vs-FDR curve.
scores/               cached discriminative scores (.npy) — not tracked; see "Data".
pyproject.toml        package metadata / install config.
requirements.txt      runtime dependencies (numpy, scipy, statsmodels).
```
