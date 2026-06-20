# TNA — Test Null Adjustment

**Reliable false discovery rate control in classification problems under distribution shift.**

Reference implementation for

> A. Borevskiy and A. Kertesz-Farkas. *Toward reliable false discovery rate control in
> classification problems under distribution shift.*

Classifier predictions are often filtered with a controlled **false discovery rate (FDR)**
using empirical p-values and the Benjamini–Hochberg procedure. Under **distribution shift**
(domain shift, batch effects, or a change in class proportions) the test score distribution
no longer matches the training one, the empirical p-values become invalid, and FDR control
silently breaks. **Test Null Adjustment (TNA)** is a simple, model-agnostic, fully
data-driven method that recalibrates the test scores in the 1-D score space so that the test
**null** distribution is realigned to the training null — restoring valid p-values and
reliable FDR control without any domain-specific assumptions.

---

## Install

```bash
pip install -e .            # installs the `epv` package + numpy/scipy/statsmodels
```

(or just `pip install -r requirements.txt` and import `epv` from the repo root.)

---

## Quick start (Python API)

```python
from epv import TNA
from epv.data import load_scores

d = load_scores("chexpert_shift")                 # cached scores for one dataset

model = TNA().fit(d["train_scores"], d["train_labels"])
q = model.qvalues(d["test_scores"])               # FDR-controlled q-values

trusted = d["test_scores"][q <= 0.10]             # discoveries at 10% FDR
print(model.pi0_)                                  # estimated null proportion
```

The estimator is `fit`/`transform` style:

```python
adjusted = model.transform(test_scores)            # recalibrated scores
pvals    = model.pvalues(test_scores)              # valid empirical p-values
```

### Command-line example

```bash
python examples/run_example.py --dataset chexpert_shift
python examples/run_example.py --dataset pcam --pi0 pounds --alpha 0.1
```

---

## Choosing the π₀ estimator

The null proportion π₀ is estimated from the data; the estimator is pluggable
(default **Storey**):

```python
TNA(pi0_method="storey")     # default — Storey & Tibshirani (2003)
TNA(pi0_method="pounds")     # Pounds & Cheng (2003): 2 * mean(p)
TNA(pi0_method="hist")       # right-tail histogram density ("Last hist")
TNA(pi0_method="nettleton")  # Nettleton / Mosig iterative histogram
TNA(pi0_method="jiang")      # Jiang & Doerge (2008) tail average
```

```python
from epv import estimate_pi0
estimate_pi0(pvalues, method="pounds")
```

These reproduce the corresponding columns of the manuscript's π₀ table closely
(e.g. CheXpert: `pounds`→0.809, `hist`→0.763, `nettleton`→0.786 vs table 0.785).
The remaining table columns — `slim` (Wang 2011) and `meinshausen` — are available
in the R packages `cp4p` / `FDRestimation` and can be added behind the same
interface in `epv/pi0.py`.

---

## Repository structure

```
epv/                  the package
├── __init__.py       public API
├── core.py           TNA estimator (fit / transform / qvalues)
├── stats.py          empirical p-values, BH q-values, ground truth
├── pi0.py            pluggable π₀ estimators (Storey default)
└── data.py           dataset loaders + model-scoring scaffolds
examples/
└── run_example.py    minimal end-to-end example
scores/               cached discriminative scores (.npy) — see "Data"
pyproject.toml · requirements.txt · LICENSE
```

---

## Data

`epv.data.load_scores(name)` loads **cached model output scores** (so the CNNs do not
have to be re-run). The `_0`/`_1` files are the negative/positive label splits (labels are
used only for ground-truth evaluation, never by the method).

```python
from epv.data import load_scores, available
available()                       # datasets whose score files are present
d = load_scores("tissuenet_shift")
```

| dataset name | source | model |
|---|---|---|
| `chexpert[_shift]` | CheXpert chest X-ray ("Effusion") | DenseNet121 (torchxrayvision) |
| `pcam[_shift]` | PatchCamelyon histopathology | ResNet34 (TIA toolbox) |
| `tissuenet[_shift]` | TissueNet cell segmentation | Cellpose |

The `_shift` variants use a class-rebalanced test set (class-distribution shift).
The score files are **not tracked in git** (size); host them with the release / on
Zenodo and place them under `scores/`. To regenerate from raw images, see the
`score_*` scaffolds in `epv/data.py` and the notebooks.

---

## Citation

```bibtex
@article{borevskiy_tna,
  title   = {Toward reliable false discovery rate control in classification
             problems under distribution shift},
  author  = {Borevskiy, Andrey and Kertesz-Farkas, Attila},
  journal = {PLOS ONE},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
