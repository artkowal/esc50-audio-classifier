"""Zestawienie wszystkich treningów: tabela wyników i wspólny wykres krzywych walidacji.

Czyta docs/wyniki/*/podsumowanie.json i historia.csv (tworzone przez trening.py).
Wynik:
    docs/wyniki/zestawienie.md                         tabela do dokumentacji
    docs/figures/trening/porownanie_walidacja.png      accuracy i strata walidacyjna wszystkich eksperymentów
Uruchomienie: python src/zestawienie.py [nazwa1 nazwa2 ...]   (bez nazw: wszystkie eksperymenty)
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dane import ROOT

RESULTS_DIR = ROOT / "docs" / "wyniki"
FIG_PATH = ROOT / "docs" / "figures" / "trening" / "porownanie_walidacja.png"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]          # maks. 3 serie na jednym wykresie (czytelność)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(names):
    rows, histories = [], {}
    dirs = [RESULTS_DIR / n for n in names] if names else sorted(p.parent for p in RESULTS_DIR.glob("*/podsumowanie.json"))
    for d in dirs:
        s = json.loads((d / "podsumowanie.json").read_text())
        rows.append({
            "Eksperyment": s["nazwa"],
            "Model": s["model"].upper(),
            "Augmentacja": "tak" if s["augmentacja"] else "nie",
            "Parametry": f"{s['parametry']:,}".replace(",", " "),
            "Najlepsza epoka": f"{s['najlepsza_epoka']} / {s['liczba_epok']}",
            "Val acc": f"{s['val_acc']:.3f}",
            "Val F1 macro": f"{s['val_f1_macro']:.3f}",
            "Val strata": f"{s['val_loss']:.3f}",
            "Train acc (najl. epoka)": f"{s['train_acc_w_najlepszej_epoce']:.3f}",
            "Czas epoki [s]": f"{s['sredni_czas_epoki_trening_s']:.1f}",
            "Łącznie [min]": f"{s['laczny_czas_s'] / 60:.1f}",
        })
        histories[s["nazwa"]] = pd.read_csv(d / "historia.csv")
    return pd.DataFrame(rows), histories


def to_markdown(df):
    """Tabela pandas -> tabela Markdown (bez dodatkowych bibliotek)."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor="white")
    for color, (name, h) in zip(COLORS, list(histories.items())[:3]):
        axes[0].plot(h["epoch"], h["val_acc"], color=color, linewidth=2, label=name)
        axes[1].plot(h["epoch"], h["val_loss"], color=color, linewidth=2, label=name)
    for ax, title in zip(axes, ["Accuracy na walidacji", "Strata na walidacji"]):
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("epoka", fontsize=9, color=MUTED)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    table, histories = load(sys.argv[1:])
    md = to_markdown(table)
    (RESULTS_DIR / "zestawienie.md").write_text("# Zestawienie treningów (walidacja)\n\n" + md + "\n")
    plot(histories, FIG_PATH)
    print(md)
    print(f"\nZapisano: {RESULTS_DIR / 'zestawienie.md'}\n          {FIG_PATH}")
