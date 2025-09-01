# analyzer/hello_model.py
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import transforms, models
import matplotlib.pyplot as plt
import yaml
import csv
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for saving figures


# ---------- tiny dataset that reads your manifest.csv ----------
class ManifestImageDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, transform=None):
        self.items = []  # (filepath, label_str)
        with open(manifest_csv, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)  # filepath,label,split
            for row in r:
                if len(row) < 2:
                    continue
                fp, label = row[0], row[1]
                self.items.append((fp, label))
        # build stable class list & mapping
        labels = sorted({lbl for _, lbl in self.items})
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        self.classes = labels
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fp, label = self.items[idx]
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as e:
            # fall back to a black image if unreadable (rare)
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[label]


# ---------- transforms preset ----------
def build_transforms(preset: str):
    if preset == "imagenet_224":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    raise ValueError(f"Unknown transforms preset: {preset}")


# ---------- model loader (ResNet-18,  with pretrained features if available) ----------
def build_model(num_classes: int, device: torch.device):
    try:
        # Newer torchvision API
        weights = models.ResNet18_Weights.DEFAULT  # type: ignore[attr-defined]
        model = models.resnet18(weights=weights)
    except Exception:
        # Fallback for older versions
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    model.eval()
    return model


# ---------- plotting ----------
def save_confusion_figure(cm: np.ndarray, class_names, out_png: Path):
    # row-normalized for readability
    with np.errstate(invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Pred")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(prog="hello-model")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    # -------- read config --------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    manifest_path = Path(cfg["manifest_path"]).expanduser()
    transforms_preset = cfg.get("transforms", "imagenet_224")
    device_str = cfg.get("device", "cpu")
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    out_prefix = cfg.get("out_prefix", "m0_run")

    # output locations
    artifacts_dir = Path("./artifacts")
    figures_dir = Path("./figs") if Path("./figs").exists() else Path("./figures")
    reports_dir = Path("./reports")
    for d in (artifacts_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -------- dataset & loader --------
    tfm = build_transforms(transforms_preset)
    ds = ManifestImageDataset(str(manifest_path), transform=tfm)
    class_names = ds.classes
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # -------- model --------
    device = torch.device(device_str)
    model = build_model(num_classes=len(class_names), device=device)

    # -------- inference --------
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])

    # -------- metrics & artifacts --------
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))) if y_true.size else np.zeros((len(class_names), len(class_names)), dtype=int)

    # save arrays
    np.save(artifacts_dir / f"{out_prefix}_y_true.npy", y_true)
    np.save(artifacts_dir / f"{out_prefix}_y_pred.npy", y_pred)
    np.save(artifacts_dir / f"{out_prefix}_confusion.npy", cm)

    # save confusion figure
    fig_path = figures_dir / f"{out_prefix}_confusion.png"
    save_confusion_figure(cm, class_names, fig_path)

    # -------- report --------
    per_class_counts = {c: int(np.sum(y_true == i)) for i, c in enumerate(class_names)}
    report = {
        "summary": {
            "num_samples": int(y_true.size),
            "num_classes": len(class_names),
            "accuracy": acc,
        },
        "classes": class_names,
        "per_class_counts": per_class_counts,
        "artifacts": {
            "y_true": str((artifacts_dir / f"{out_prefix}_y_true.npy").resolve()),
            "y_pred": str((artifacts_dir / f"{out_prefix}_y_pred.npy").resolve()),
            "confusion": str((artifacts_dir / f"{out_prefix}_confusion.npy").resolve()),
            "confusion_png": str(fig_path.resolve()),
        },
        "config_snapshot": cfg,
    }
    with open(reports_dir / f"{out_prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[M0] Done. Accuracy={acc:.4f}")
    print(f"Report: {reports_dir / f'{out_prefix}_report.json'}")
    print(f"Confusion fig: {fig_path}")


if __name__ == "__main__":
    sys.exit(main())
