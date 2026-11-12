"""Zajęcia 8: predykcje wsadowe na zbiorze testowym (fold 5) i metryki.

Użycie (po wyborze ostatecznych modeli):
    python src/ewaluacja.py models/cnn_bezaug_lr0.001_cosine_val4_test5.pt
    python src/ewaluacja.py models/<ostateczny_gru>.pt

Dla każdego modelu zapisuje:
    docs/wyniki/test_<nazwa>/predykcje.csv     plik, prawdziwa klasa, przewidziana, pewność, 50 prawdopodobieństw
    docs/wyniki/test_<nazwa>/metryki.json      accuracy, Precision/Recall/F1 (macro), ROC AUC, PR AUC, czasy
    docs/wyniki/test_<nazwa>/raport_klas.csv   Precision/Recall/F1 dla każdej z 50 klas
    docs/wyniki/test_<nazwa>/pomylki.csv       najczęściej mylone pary klas
    docs/wyniki/test_<nazwa>/bledy.csv         błędne predykcje (od najbardziej pewnych)
    docs/figures/test/<nazwa>_macierz.png      macierz pomyłek 50x50
    docs/figures/test/<nazwa>_roc_pr.png       krzywe ROC i PR (uśrednione po wszystkich klasach)
Zbiór testowy jest używany tylko tutaj, raz, dla modeli wybranych wcześniej na walidacji.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from dane import ROOT

RESULTS_DIR = ROOT / "docs" / "wyniki"
FIG_DIR = ROOT / "docs" / "figures" / "test"
N_CLASSES = 50
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------- Model i predykcje ----------

def load_model(path, device):
    """Odtwarza model z checkpointu zapisanego przez trening.py."""
    import torch
    from modele import CNNModel, GRUModel

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if ckpt["model_name"] == "cnn":      # statystyki normalizacji wczytają się z wag (bufory LogMel)
        model = CNNModel(dropout=cfg.get("dropout", 0.3), use_specaug=cfg.get("augment", True))
    else:
        model = GRUModel(dropout=cfg.get("dropout", 0.3), gru_dropout=cfg.get("gru_dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def predict(model, loader, device):
    """Predykcje wsadowe (batch 32): prawdopodobieństwa [N, 50] i czasy."""
    import torch

    probs, labels = [], []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for wave, y in loader:
            logits = model(wave.to(device, non_blocking=True))
            probs.append(torch.softmax(logits, dim=1).cpu())    # softmax: logity -> prawdopodobieństwa
            labels.append(y)
    if device == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return probs, labels, {"czas_calkowity_s": round(total, 3),
                           "czas_na_nagranie_ms": round(1000 * total / len(labels), 3),
                           "czas_na_batch_ms": round(1000 * total / len(loader), 2)}


# ---------- Metryki i wykresy (bez PyTorcha) ----------

def compute_metrics(y, probs, class_names):
    pred = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    y_bin = label_binarize(y, classes=range(N_CLASSES))
    metrics = {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision_macro": round(p, 4), "recall_macro": round(r, 4), "f1_macro": round(f1, 4),
        "roc_auc_macro": round(roc_auc_score(y_bin, probs, average="macro"), 4),
        "pr_auc_macro": round(average_precision_score(y_bin, probs, average="macro"), 4),
        "top3_accuracy": round(float(np.mean([t in np.argsort(-pr)[:3] for t, pr in zip(y, probs)])), 4),
    }
    report = pd.DataFrame(classification_report(y, pred, labels=range(N_CLASSES), target_names=class_names,
                                                output_dict=True, zero_division=0)).T.iloc[:N_CLASSES]
    return pred, metrics, report.round(3)


def top_confusions(cm, class_names, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(class_names[i], class_names[j], int(off[i, j])) for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None), off.shape))]
    return pd.DataFrame([p for p in pairs if p[2] > 0][:k], columns=["prawdziwa", "przewidziana", "liczba"])


def plot_confusion(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(13, 12), facecolor="white")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(N_CLASSES), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(N_CLASSES), class_names, fontsize=6)
    ax.set_xlabel("przewidziana klasa", fontsize=9, color=MUTED)
    ax.set_ylabel("prawdziwa klasa", fontsize=9, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01).set_label("liczba nagrań", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(y, probs, metrics, title, path):
    """Krzywe uśrednione „micro” (wszystkie klasy razem); w legendzie AUC uśrednione „macro”."""
    y_bin = label_binarize(y, classes=range(N_CLASSES))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    prec, rec, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")
    axes[0].plot(fpr, tpr, color="#2a78d6", linewidth=2, label=f"ROC AUC (macro) = {metrics['roc_auc_macro']:.3f}")
    axes[0].plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="losowe zgadywanie")
    axes[0].set_xlabel("odsetek fałszywych alarmów (FPR)", fontsize=9, color=MUTED)
    axes[0].set_ylabel("czułość (TPR)", fontsize=9, color=MUTED)
    axes[0].set_title("Krzywa ROC", loc="left", fontsize=10, color=INK)
    axes[1].plot(rec, prec, color="#2a78d6", linewidth=2, label=f"PR AUC (macro) = {metrics['pr_auc_macro']:.3f}")
    axes[1].axhline(1 / N_CLASSES, color=MUTED, linestyle="--", linewidth=1, label="losowe zgadywanie")
    axes[1].set_xlabel("Recall", fontsize=9, color=MUTED)
    axes[1].set_ylabel("Precision", fontsize=9, color=MUTED)
    axes[1].set_title("Krzywa Precision-Recall", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.legend(frameon=False, fontsize=9, loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_all(name, files, y, probs, class_names, timing, extra):
    out = RESULTS_DIR / f"test_{name}"
    out.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pred, metrics, report = compute_metrics(y, probs, class_names)
    cm = confusion_matrix(y, pred, labels=range(N_CLASSES))

    df = pd.DataFrame({"plik": files, "prawdziwa": [class_names[t] for t in y],
                       "przewidziana": [class_names[t] for t in pred], "pewnosc": probs.max(axis=1).round(4)})
    df = pd.concat([df, pd.DataFrame(probs.round(5), columns=[f"p_{c}" for c in class_names])], axis=1)
    df.to_csv(out / "predykcje.csv", index=False)
    report.to_csv(out / "raport_klas.csv")
    conf = top_confusions(cm, class_names)
    conf.to_csv(out / "pomylki.csv", index=False)
    errors = df[df["prawdziwa"] != df["przewidziana"]][["plik", "prawdziwa", "przewidziana", "pewnosc"]]
    errors.sort_values("pewnosc", ascending=False).to_csv(out / "bledy.csv", index=False)

    (out / "metryki.json").write_text(json.dumps({**metrics, **timing, **extra}, indent=2, ensure_ascii=False))
    plot_confusion(cm, class_names, f"{name}: macierz pomyłek na teście (fold 5)", FIG_DIR / f"{name}_macierz.png")
    plot_roc_pr(y, probs, metrics, f"{name}: zbiór testowy (fold 5)", FIG_DIR / f"{name}_roc_pr.png")
    return metrics, report, conf, errors


if __name__ == "__main__":
    import torch
    from dane import load_metadata
    from potok import make_loaders

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpointy", nargs="+", type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_meta = load_metadata()
    class_names = df_meta.drop_duplicates("target").sort_values("target")["category"].tolist()

    for path in args.checkpointy:
        model, cfg = load_model(path, device)
        loaders, _, idx = make_loaders(cfg["val_fold"], cfg["test_fold"], augment=False)
        files = df_meta.loc[idx["test"], "filename"].tolist()
        probs, y, timing = predict(model, loaders["test"], device)

        name = path.stem
        params = sum(p.numel() for p in model.parameters())
        extra = {"model": cfg["model"], "parametry": params, "checkpoint": str(path), "test_fold": cfg["test_fold"]}
        metrics, report, conf, errors = save_all(name, files, y, probs, class_names, timing, extra)

        print(f"\n===== {name} (test: fold {cfg['test_fold']}, {len(y)} nagrań) =====")
        for k, v in metrics.items():
            print(f"  {k:16s} {v}")
        print(f"  czas predykcji: {timing['czas_na_nagranie_ms']} ms / nagranie, {timing['czas_na_batch_ms']} ms / batch 32")
        print(f"  błędnych predykcji: {len(errors)} z {len(y)}")
        print("  najsłabsze klasy (F1):", ", ".join(f"{c} {v:.2f}" for c, v in report["f1-score"].nsmallest(5).items()))
        print("  najczęstsze pomyłki:", "; ".join(f"{r.prawdziwa} -> {r.przewidziana} ({r.liczba})" for r in conf.head(5).itertuples()))
        print(f"  zapisano: {RESULTS_DIR / f'test_{name}'}, {FIG_DIR}")
