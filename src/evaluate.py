import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

try:
    from .dataset import build_loaders
    from .model import build_model
except ImportError:
    from dataset import build_loaders
    from model import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    probs, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
        labels.extend(lbls.numpy())
    return np.array(probs), np.array(labels)


def find_optimal_threshold(y_true, y_prob):
    """Youden's J statistic: maximise sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx]


def evaluate_test_set(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = build_loaders(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["model_save"], map_location=device))

    y_prob, y_true = get_predictions(model, test_loader, device)

    thr = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== Test Set Results (N={len(y_true)}) ===")
    print(f"AUC-ROC      : {auc:.4f}")
    print(f"Threshold    : {thr:.4f}")
    print(f"Sensitivity  : {sens:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Pneumonia Classification")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(cfg["paths"]["roc"], dpi=150)
    plt.close()

    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("./outputs/confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "auc": auc,
        "sensitivity": sens,
        "threshold": thr,
        "precision": prec,
        "f1": f1,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    evaluate_test_set(cfg)
