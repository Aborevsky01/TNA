"""Data access for the four benchmark datasets.

Two layers:

1. ``load_scores(name)`` - load the cached discriminative scores (.npy) shipped
   in ``<repo>/scores`` (or any directory via ``scores_dir=``). This is all that
   is needed to run TNA and reproduce the FDR-control experiments.

2. ``score_*`` scaffolds - regenerate scores from the raw images + pretrained
   models. These require the raw datasets and extra dependencies (torch,
   torchxrayvision, tiatoolbox, cellpose) and are provided for completeness.
"""
from __future__ import annotations
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_DIR = os.path.join(ROOT, "scores")

# name -> (train_neg, train_pos, test_neg, test_pos, null_hist_file)
DATASETS = {
    "chexpert":        ("xray_x_0", "xray_x_1", "xray_t_0", "xray_t_1", "xray_test_neg_AKF_hist.npy"),
    "chexpert_shift":  ("xray_x_0", "xray_x_1", "chx_reb_t_0", "chx_reb_t_1", "xray reb_test_neg_AKF_hist.npy"),
    "pcam":            ("pcam_x_0", "pcam_x_1", "pcam_t_0", "pcam_t_1", "pcam_test_neg_AKF_hist.npy"),
    "pcam_shift":      ("pcam_x_0", "pcam_x_1", "pcam_reb_t_0", "pcam_reb_t_1", "pcam_reb_test_neg_AKF_hist.npy"),
    "tissuenet":       ("cell_x_0", "cell_x_1", "cell_t_0", "cell_t_1", "cell_test_neg_AKF_hist.npy"),
    "tissuenet_shift": ("cell_x_0", "cell_x_1", "cell_reb_t_0", "cell_reb_t_1", "cell_reb_test_neg_AKF_hist.npy"),
}

# Pretrained models used to produce the scores (for the score_* scaffolds).
MODELS = {
    "chexpert":  "DenseNet121 'densenet121-res224-all' (torchxrayvision), target 'Effusion'",
    "pcam":      "ResNet34 'resnet34-pcam' (TIA toolbox)",
    "tissuenet": "Cellpose (fine-tuned)",
    "bcss":      "FCN-ResNet50 'fcn_resnet50_unet-bcss' (TIA toolbox), multi-class",
}


def load_scores(name, scores_dir=None):
    """Return a dict with train/test scores, binary labels and the null histogram.

    Keys: train_scores, train_labels, test_scores, test_labels, null_hist (or None).
    The ``_0``/``_1`` files are the negative/positive label splits of the cached
    model outputs (labels are used only for ground-truth evaluation).
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    d = scores_dir or SCORES_DIR
    x0n, x1n, t0n, t1n, nhf = DATASETS[name]

    def L(fn):
        return np.load(os.path.join(d, fn)).astype(np.float32)

    x0, x1 = L(x0n + ".npy"), L(x1n + ".npy")
    t0, t1 = L(t0n + ".npy"), L(t1n + ".npy")
    nh_path = os.path.join(d, nhf)
    null_hist = np.load(nh_path) if os.path.exists(nh_path) else None
    return {
        "train_scores": np.concatenate([x0, x1]),
        "train_labels": np.concatenate([np.zeros(x0.size), np.ones(x1.size)]),
        "test_scores":  np.concatenate([t0, t1]),
        "test_labels":  np.concatenate([np.zeros(t0.size), np.ones(t1.size)]),
        "null_hist":    null_hist,
    }


def available(scores_dir=None):
    """List datasets whose score files are present."""
    d = scores_dir or SCORES_DIR
    out = []
    for name, files in DATASETS.items():
        if all(os.path.exists(os.path.join(d, f + ".npy")) for f in files[:4]):
            out.append(name)
    return out


# --------------------------------------------------------------------------- #
# Score-generation scaffolds (require raw data + extra dependencies).
# --------------------------------------------------------------------------- #
def score_chexpert(imgpath, csvpath, weights_path, target="Effusion", batch_size=32):
    """Score CheXpert with a fine-tuned DenseNet121. Returns (scores, labels)."""
    import torch, torchxrayvision as xrv, torchvision, sklearn.model_selection as ms
    transform = torchvision.transforms.Compose(
        [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    ds = xrv.datasets.CheX_Dataset(imgpath=imgpath, csvpath=csvpath, transform=transform)
    ds.labels = np.nan_to_num(ds.labels, 0)
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.op_threshs = None
    model.classifier = torch.nn.Linear(1024, 1)
    model.load_state_dict(torch.load(weights_path)); model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["img"]).squeeze(-1)
            scores.extend(out.cpu().numpy())
            labels.extend(batch["lab"][:, ds.pathologies.index(target)].cpu().numpy())
    return np.asarray(scores), np.asarray(labels)


def score_pcam(*args, **kwargs):
    raise NotImplementedError(
        "PCam scoring uses the TIA-toolbox 'resnet34-pcam' model. See notebooks/histopathology.ipynb.")


def score_tissuenet(*args, **kwargs):
    raise NotImplementedError(
        "TissueNet scoring uses a fine-tuned Cellpose model. See notebooks/cells_segment.ipynb.")


def score_bcss(*args, **kwargs):
    raise NotImplementedError(
        "BCSS is multi-class (one-vs-rest), scored with TIA-toolbox 'fcn_resnet50_unet-bcss'. "
        "See notebooks/breast_cancer.ipynb.")
