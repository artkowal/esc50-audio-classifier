"""Podsumowanie 5-krotnej walidacji krzyżowej: średnia ± odchylenie metryk testowych z 5 foldów.

Najpierw ewaluacja wszystkich checkpointów CV (każdy na swoim foldzie testowym):
    python src/ewaluacja.py models/{cnn_bezaug,gru_aug}_lr0.001_cosine_{val5_test1,val1_test2,val2_test3,val3_test4,val4_test5}.pt
Potem:
    python src/cv_podsumowanie.py cnn_bezaug_lr0.001_cosine gru_aug_lr0.001_cosine
Wynik: tabela w terminalu i w docs/wyniki/walidacja_krzyzowa.md
"""
import argparse
import json
import statistics as st

from dane import ROOT

RESULTS_DIR = ROOT / "docs" / "wyniki"
METRICS = [("accuracy", "Accuracy"), ("f1_macro", "F1 macro"), ("precision_macro", "Precision macro"),
           ("recall_macro", "Recall macro"), ("roc_auc_macro", "ROC AUC"), ("pr_auc_macro", "PR AUC"),
           ("top3_accuracy", "Top-3 accuracy")]


def val_fold_for(test_fold):
    """Schemat CV: walidacja = fold poprzedzający test (dla testu 1: fold 5)."""
    return 5 if test_fold == 1 else test_fold - 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("warianty", nargs="+")
    args = p.parse_args()

    lines = []
    summary = {}
    for w in args.warianty:
        per_fold = {}
        for k in range(1, 6):
            path = RESULTS_DIR / f"test_{w}_val{val_fold_for(k)}_test{k}" / "metryki.json"
            per_fold[k] = json.loads(path.read_text())
        summary[w] = per_fold
        lines += [f"## {w}", "", "| Fold testowy | " + " | ".join(n for _, n in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for k, m in per_fold.items():
            lines.append(f"| {k} | " + " | ".join(f"{m[key]:.3f}" for key, _ in METRICS) + " |")
        lines.append("| **średnia ± odch.** | " + " | ".join(
            f"**{st.mean(m[key] for m in per_fold.values()):.3f} ± {st.stdev(m[key] for m in per_fold.values()):.3f}**"
            for key, _ in METRICS) + " |")
        lines.append("")

    md = "\n".join(lines)
    out = RESULTS_DIR / "walidacja_krzyzowa.md"
    out.write_text("# 5-krotna walidacja krzyżowa (oficjalne foldy ESC-50)\n\n"
                   "W iteracji k: test = fold k, walidacja = fold poprzedzający, trening = pozostałe 3 foldy.\n\n" + md)
    print(md)
    print(f"Zapisano: {out}")
