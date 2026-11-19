"""Zajęcia 9: porównanie ostatecznych modeli CNN i GRU.

1. Koszt obliczeniowy: parametry, rozmiar pliku, czas epoki, czas predykcji (GPU i CPU), pamięć GPU.
2. Przestrzeń cech wyciągniętych z sieci (wejście ostatniej warstwy) na zbiorze testowym:
   mapa t-SNE + miara rozdzielenia klas (silhouette) w porównaniu z cechami MFCC z EDA.
3. Tabela z wynikami z literatury.

Użycie:
    python src/porownanie.py
Wynik:
    docs/wyniki/porownanie_modeli.md
    docs/figures/porownanie/tsne_cechy_sieci.png
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from dane import GROUPS, ROOT

RESULTS_DIR = ROOT / "docs" / "wyniki"
FIG_DIR = ROOT / "docs" / "figures" / "porownanie"
MODELS = {"CNN": "cnn_aug_lr0.001_cosine", "GRU": "gru_aug_lr0.001_cosine"}   # ostateczne modele
FINAL = "_val4_test5"
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, BACKDROP, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d4", "#ffffff"
LITERATURE = [("Prosta CNN na spektrogramie (Piczak, 2015)", "64,5%"),
              ("Sieci na surowym sygnale (zależnie od głębokości)", "51,1–68,5%"),
              ("Człowiek", "81,3%")]


# ---------- Część bez PyTorcha: wykres i miary ----------

def separation(features, labels):
    """Silhouette (−1..1) po standaryzacji: im wyżej, tym lepiej nagrania jednej klasy trzymają się razem."""
    return float(silhouette_score(StandardScaler().fit_transform(features), labels, metric="cosine"))


def plot_tsne(maps, groups, path):
    """Siatka: wiersz = reprezentacja, kolumna = grupa klas (podświetlona), reszta szara."""
    fig, axes = plt.subplots(len(maps), 5, figsize=(16, 3.3 * len(maps)), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for r, (name, xy) in enumerate(maps.items()):
        for c, g in enumerate(GROUPS):
            ax = axes[r, c]
            mask = groups == g
            ax.scatter(xy[~mask, 0], xy[~mask, 1], s=5, color=BACKDROP, linewidths=0)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=10, color=GROUP_COLORS[c], edgecolors=SURFACE, linewidths=0.3)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GRID)
            if r == 0:
                ax.set_title(g, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, color=INK)
    fig.suptitle("t-SNE na zbiorze testowym (fold 5, 400 nagrań): cechy MFCC vs cechy wyuczone przez sieci",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(cost, sep, cv_acc):
    lines = ["# Porównanie modeli CNN i GRU", "", "## Koszt obliczeniowy", "",
             "| | CNN | GRU |", "|---|---|---|"]
    labels = [("parametry", "Parametry"), ("rozmiar_mb", "Rozmiar checkpointu [MB]"),
              ("czas_epoki_s", "Czas epoki treningu [s]"), ("laczny_czas_min", "Łączny czas treningu [min]"),
              ("gpu_ms_na_nagranie", "Predykcja GPU, batch 32 [ms / nagranie]"),
              ("cpu_ms_na_nagranie", "Predykcja CPU, pojedyncze nagranie [ms]"),
              ("pamiec_trening_gb", "Pamięć GPU: krok treningu, batch 32 [GB]"),
              ("pamiec_predykcja_gb", "Pamięć GPU: predykcja, batch 32 [GB]")]
    for key, label in labels:
        lines.append(f"| {label} | {cost['CNN'][key]} | {cost['GRU'][key]} |")
    lines += ["", "## Rozdzielenie klas w przestrzeni cech (test, silhouette, cosinus)", "",
              "| Reprezentacja | Silhouette (50 klas) |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in sep.items()]
    lines += ["", "Silhouette: od −1 do 1; wyżej = nagrania tej samej klasy leżą bliżej siebie niż innych klas.",
              "", "## Porównanie z literaturą (modele uczone tylko na ESC-50)", "",
              "| Model | Accuracy |", "|---|---|"]
    lines += [f"| {m} | {a} |" for m, a in LITERATURE]
    lines += [f"| **Nasz CNN** (5-krotna walidacja krzyżowa) | **{cv_acc['CNN']}** |",
              f"| **Nasz GRU** (5-krotna walidacja krzyżowa) | **{cv_acc['GRU']}** |"]
    out = RESULTS_DIR / "porownanie_modeli.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out, "\n".join(lines)


def cv_accuracy(variant):
    """Średnia ± odchylenie accuracy z 5-krotnej walidacji krzyżowej (pliki z ewaluacja.py)."""
    import statistics as st
    accs = []
    for k in range(1, 6):
        path = RESULTS_DIR / f"test_{variant}_val{5 if k == 1 else k - 1}_test{k}" / "metryki.json"
        if not path.exists():
            return "brak wyników CV"
        accs.append(json.loads(path.read_text())["accuracy"])
    return f"{st.mean(accs):.3f} ± {st.stdev(accs):.3f}".replace(".", ",")


# ---------- Część z PyTorchem ----------

def embeddings(model, loader, device):
    """Wejście ostatniej warstwy (głowy) = wektor cech wyuczony przez sieć: CNN 256, GRU 192 liczby."""
    import torch
    feats = []
    hook = model.head.register_forward_pre_hook(lambda m, inp: feats.append(inp[0].flatten(1).cpu()))
    with torch.no_grad():
        for wave, _ in loader:
            model(wave.to(device))
    hook.remove()
    return torch.cat(feats).numpy()


def measure_cost(model, loader, device, path, summary):
    import torch
    wave, y = next(iter(loader))
    cost = {"parametry": f"{sum(p.numel() for p in model.parameters()):,}".replace(",", " "),
            "rozmiar_mb": round(path.stat().st_size / 1e6, 2),
            "czas_epoki_s": summary["sredni_czas_epoki_trening_s"],
            "laczny_czas_min": round(summary["laczny_czas_s"] / 60, 1)}

    if device != "cuda":                         # bez karty graficznej: pomiary GPU pomijamy
        cost.update(gpu_ms_na_nagranie="—", pamiec_predykcja_gb="—", pamiec_trening_gb="—")
        return measure_cpu(model, wave, cost)

    # Predykcja na GPU: rozgrzewka, potem średnia z 3 przebiegów przez cały zbiór testowy
    with torch.no_grad():
        model(wave.to(device))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            for w, _ in loader:
                model(w.to(device))
        torch.cuda.synchronize()
    cost["gpu_ms_na_nagranie"] = round(1000 * (time.perf_counter() - t0) / (3 * len(loader.dataset)), 3)

    # Pamięć GPU: predykcja i krok treningu
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(wave.to(device))
    cost["pamiec_predykcja_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    torch.nn.functional.cross_entropy(model(wave.to(device)), y.to(device)).backward()
    cost["pamiec_trening_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    model.zero_grad(set_to_none=True)
    model.eval()
    cost = measure_cpu(model, wave, cost)
    model.to(device)
    return cost


def measure_cpu(model, wave, cost):
    """Predykcja na CPU, po jednym nagraniu (jak w aplikacji bez karty graficznej)."""
    import torch
    cpu_model = model.to("cpu").eval()
    with torch.no_grad():
        cpu_model(wave[:1])
        t0 = time.perf_counter()
        for i in range(20):
            cpu_model(wave[i:i + 1])
    cost["cpu_ms_na_nagranie"] = round(1000 * (time.perf_counter() - t0) / 20, 1)
    return cost


if __name__ == "__main__":
    import torch
    from dane import load_metadata
    from ewaluacja import load_model
    from potok import make_loaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("UWAGA: karta graficzna niedostępna, liczę na CPU (pomiary GPU zostaną pominięte).")
    torch.manual_seed(0)
    df = load_metadata()
    loaders, _, idx = make_loaders(4, 5, augment=False)
    groups = df.loc[idx["test"], "group"].to_numpy()
    labels = df.loc[idx["test"], "target"].to_numpy()

    feats, cost = {}, {}
    for name, variant in MODELS.items():
        path = ROOT / "models" / f"{variant}{FINAL}.pt"
        model, _ = load_model(path, device)
        summary = json.loads((RESULTS_DIR / f"{variant}{FINAL}" / "podsumowanie.json").read_text())
        feats[f"{name} (cechy sieci)"] = embeddings(model, loaders["test"], device)
        cost[name] = measure_cost(model, loaders["test"], device, path, summary)
        print(f"{name}: {cost[name]}")

    # Cechy MFCC z EDA (cache z eda_wizualizacje.py), dla tych samych 400 nagrań testowych
    cache = ROOT / "data" / "cache" / "eda_cechy.npz"
    if cache.exists():
        feats = {"MFCC (EDA)": np.load(cache)["mfcc"][idx["test"]], **feats}

    sep = {k: separation(v, labels) for k, v in feats.items()}
    maps = {k: TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(
        StandardScaler().fit_transform(v)) for k, v in feats.items()}
    plot_tsne(maps, groups, FIG_DIR / "tsne_cechy_sieci.png")

    cv_acc = {name: cv_accuracy(variant) for name, variant in MODELS.items()}
    out, md = write_report(cost, sep, cv_acc)
    print("\n" + md)
    print(f"\nZapisano: {out}\n          {FIG_DIR / 'tsne_cechy_sieci.png'}")
