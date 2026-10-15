"""EDA, krok 2: rozkład etykiet (klasy, grupy, podział na trening/walidację/test).

Wynik:
- docs/eda/rozklad_klas.csv: liczebność każdej klasy w każdym zbiorze,
- docs/figures/eda/rozklad_klas.png: wykres rozkładu,
- podsumowanie w terminalu (w tym kontrola: 8 nagrań każdej klasy w każdym foldzie).
"""
import matplotlib

matplotlib.use("Agg")                      # zapis do pliku, bez otwierania okna
import matplotlib.pyplot as plt
import pandas as pd

from dane import GROUPS, ROOT, load_metadata

TABLE_DIR = ROOT / "docs" / "eda"
FIG_DIR = ROOT / "docs" / "figures" / "eda"

SPLITS = ["trening", "walidacja", "test"]
COLORS = {"trening": "#2a78d6", "walidacja": "#eb6834", "test": "#1baf7a"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def class_table(df):
    """Tabela: klasa x zbiór (liczba nagrań) + suma i udział procentowy."""
    t = df.pivot_table(index=["target", "category", "group"], columns="split",
                       values="filename", aggfunc="count", fill_value=0)[SPLITS]
    t["razem"] = t.sum(axis=1)
    t["udzial_%"] = (100 * t["razem"] / t["razem"].sum()).round(2)
    return t.reset_index()


def plot_distribution(t, path):
    """Poziomy wykres skumulowany: każda klasa podzielona na trening/walidację/test."""
    t = t.iloc[::-1].reset_index(drop=True)          # klasa 0 na górze
    fig, ax = plt.subplots(figsize=(8, 13), facecolor=SURFACE)
    left = pd.Series(0, index=t.index)
    for split in SPLITS:
        ax.barh(t.index, t[split], left=left, height=0.72, color=COLORS[split],
                edgecolor=SURFACE, linewidth=1.5, label=split)
        left += t[split]

    ax.set_yticks(t.index, t["category"], fontsize=8, color=INK)
    ax.set_xlim(0, 44)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    # Etykiety grup i linie podziału między grupami
    for g, rows in t.groupby("group", sort=False):
        ax.text(42.3, rows.index.to_numpy().mean(), g, rotation=-90, va="center", ha="center",
                fontsize=9, color=INK_MUTED)
        ax.axhline(rows.index.min() - 0.5, color=GRID, linewidth=0.8)

    ax.set_xlabel("Liczba nagrań", color=INK_MUTED, fontsize=9)
    ax.set_title("ESC-50: liczba nagrań w każdej z 50 klas (po 40, w tym 24 / 8 / 8)",
                 loc="left", fontsize=11, color=INK, pad=24)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9,
              borderaxespad=0.2, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()

    t = class_table(df)
    t.to_csv(TABLE_DIR / "rozklad_klas.csv", index=False)
    plot_distribution(t, FIG_DIR / "rozklad_klas.png")

    print("=== Rozkład etykiet ===")
    print(f"Klas: {len(t)}, nagrań: {t['razem'].sum()}")
    print(f"Najmniej liczna klasa: {t['razem'].min()}, najliczniejsza: {t['razem'].max()}, "
          f"stosunek max/min: {t['razem'].max() / t['razem'].min():.2f}")
    print("Liczba nagrań na klasę w zbiorach:",
          {s: sorted(t[s].unique().tolist()) for s in SPLITS})

    per_fold = df.groupby(["fold", "target"]).size().unstack(fill_value=0)
    print(f"Każda klasa ma dokładnie 8 nagrań w każdym foldzie: {bool((per_fold == 8).all().all())}")

    print("\nGrupy klas (nagrania w zbiorach):")
    print(df.pivot_table(index="group", columns="split", values="filename", aggfunc="count")[SPLITS]
          .reindex(GROUPS).to_string())

    print(f"\nZapisano: {TABLE_DIR / 'rozklad_klas.csv'}")
    print(f"Zapisano: {FIG_DIR / 'rozklad_klas.png'}")