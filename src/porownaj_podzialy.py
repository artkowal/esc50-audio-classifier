"""Porównanie wariantów treningu na kilku podziałach (walidacja = fold 1, 2, 3, 4; test = fold 5 nietknięty).

Czyta docs/wyniki/<wariant>_val<v>_test5/podsumowanie.json (tworzone przez trening.py) i liczy
średnią oraz odchylenie standardowe wyników walidacji dla każdego wariantu.

Przykład:
    python src/porownaj_podzialy.py cnn_aug_lr0.001_cosine cnn_bezaug_lr0.001_cosine --wyjscie augmentacja_cnn
Wynik: tabela w terminalu i w docs/wyniki/<wyjscie>.md
"""

# python src/porownaj_podzialy.py cnn_aug_lr0.001_cosine cnn_bezaug_lr0.001_cosine --wyjscie augmentacja_cnn


import argparse
import json
import statistics as st

from dane import ROOT

RESULTS_DIR = ROOT / "docs" / "wyniki"


def load(variant, folds, test_fold):
    rows = []
    for v in folds:
        path = RESULTS_DIR / f"{variant}_val{v}_test{test_fold}" / "podsumowanie.json"
        s = json.loads(path.read_text())
        rows.append({"fold": v, "acc": s["val_acc"], "f1": s["val_f1_macro"], "loss": s["val_loss"],
                     "train_acc": s["train_acc_w_najlepszej_epoce"]})
    return rows


def mean_sd(values):
    return f"{st.mean(values):.3f} ± {st.stdev(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("warianty", nargs="+", help="nazwy eksperymentów bez końcówki _val<v>_test<t>")
    p.add_argument("--foldy", type=int, nargs="+", default=[1, 2, 3, 4], help="foldy walidacyjne")
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--wyjscie", default="porownanie_podzialow")
    args = p.parse_args()

    data = {w: load(w, args.foldy, args.test_fold) for w in args.warianty}

    lines = ["## Wyniki na każdym podziale (val acc)", "",
             "| Wariant | " + " | ".join(f"val = fold {v}" for v in args.foldy) + " |",
             "|---|" + "---|" * len(args.foldy)]
    for w, rows in data.items():
        lines.append(f"| {w} | " + " | ".join(f"{r['acc']:.3f}" for r in rows) + " |")

    lines += ["", "## Średnia ± odchylenie standardowe", "",
              "| Wariant | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Różnica trening–walidacja |",
              "|---|---|---|---|---|---|"]
    for w, rows in data.items():
        acc = [r["acc"] for r in rows]
        gap = [r["train_acc"] - r["acc"] for r in rows]
        lines.append(f"| {w} | {mean_sd(acc)} | {mean_sd([r['f1'] for r in rows])} | "
                     f"{mean_sd([r['loss'] for r in rows])} | {mean_sd([r['train_acc'] for r in rows])} | "
                     f"{mean_sd(gap)} |")

    if len(data) == 2:                           # porównanie parami: ten sam podział, dwa warianty
        (a, ra), (b, rb) = data.items()
        diffs = [y["acc"] - x["acc"] for x, y in zip(ra, rb)]
        wins = sum(d > 0 for d in diffs)
        lines += ["", f"**{b} − {a}** (val acc, ten sam podział): "
                      + ", ".join(f"{100 * d:+.1f} p.p." for d in diffs)
                      + f" | średnio {100 * st.mean(diffs):+.1f} p.p. | lepszy w {wins} z {len(diffs)} podziałów"]

    md = "\n".join(lines)
    out = RESULTS_DIR / f"{args.wyjscie}.md"
    out.write_text(f"# Porównanie wariantów na {len(args.foldy)} podziałach (test = fold {args.test_fold} nieużywany)\n\n"
                   + md + "\n")
    print(md)
    print(f"\nZapisano: {out}")
