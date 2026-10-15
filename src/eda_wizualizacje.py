"""EDA, krok 3: wizualizacje, rozkład cech, PCA / t-SNE / UMAP i nietypowe próbki.

Wynik (wykresy w docs/figures/eda/, tabele w docs/eda/):
- przyklady_sygnal_spektrogram.png: przebieg surowy (44,1 kHz) i log-mel-spektrogram po przetwarzaniu (16 kHz),
- rozklad_cech.png: głośność, „jasność” dźwięku i udział ciszy w 5 grupach klas,
- przestrzen_cech.png: PCA, t-SNE i UMAP na cechach MFCC,
- nietypowe_isolation_forest.csv: najbardziej nietypowe nagrania według Isolation Forest.
Cechy liczone raz są zapisywane w data/cache/ i używane przy kolejnych uruchomieniach.
"""
import matplotlib

matplotlib.use("Agg")
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from dane import GROUPS, ROOT, load_metadata

TABLE_DIR = ROOT / "docs" / "eda"
FIG_DIR = ROOT / "docs" / "figures" / "eda"
CACHE = ROOT / "data" / "cache" / "eda_cechy.npz"

# Parametry przetwarzania zgodne ze specyfikacją architektury
SR = 16000
N_FFT, HOP, N_MELS = 1024, 160, 128
N_MFCC = 20

EXAMPLE_CLASSES = ["dog", "rain", "coughing", "glass_breaking", "siren"]   # po jednej z każdej grupy
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_MUTED, GRID, BACKDROP, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d4", "#ffffff"


def load_16k(path):
    """Wczytanie pliku i przepróbkowanie z 44,1 kHz do 16 kHz."""
    x, sr = sf.read(path, dtype="float32")
    return librosa.resample(x, orig_sr=sr, target_sr=SR)


def log_mel(x):
    """Log-mel-spektrogram [128 pasm x 501 ramek] w dB, jak na wejściu CNN."""
    mel = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max, top_db=80)


def compute_features(df):
    """Dla każdego nagrania: wektor MFCC (średnia i odchylenie 20 współczynników = 40 liczb),
    głośność RMS [dBFS] i średni środek ciężkości widma [Hz] („jasność” dźwięku)."""
    if CACHE.exists():
        d = np.load(CACHE)
        return d["mfcc"], d["rms_db"], d["centroid"]
    mfcc, rms_db, centroid = [], [], []
    for path in tqdm(df["path"], desc="Cechy"):
        x = load_16k(path)
        m = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
        mfcc.append(np.concatenate([m.mean(axis=1), m.std(axis=1)]))
        rms_db.append(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12))
        centroid.append(librosa.feature.spectral_centroid(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP).mean())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    mfcc, rms_db, centroid = (np.array(v, dtype=float) for v in (mfcc, rms_db, centroid))
    np.savez(CACHE, mfcc=mfcc, rms_db=rms_db, centroid=centroid)
    return mfcc, rms_db, centroid


def style(ax):
    """Wspólny, spokojny wygląd osi."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_examples(df):
    """Dla 5 klas: przebieg surowy (lewa kolumna) i log-mel-spektrogram po przetwarzaniu (prawa)."""
    fig, axes = plt.subplots(len(EXAMPLE_CLASSES), 2, figsize=(12, 11), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    train = df[df["split"] == "trening"]
    for row, cls in enumerate(EXAMPLE_CLASSES):
        item = train[train["category"] == cls].iloc[0]
        raw, sr = sf.read(item["path"], dtype="float32")
        t = np.arange(len(raw)) / sr
        ax = axes[row, 0]
        ax.plot(t, raw, color=GROUP_COLORS[GROUPS.index(item["group"])], linewidth=0.5)
        ax.set_xlim(0, 5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"{cls}\n({item['group']})", fontsize=9, color=INK)
        style(ax)

        ax = axes[row, 1]
        im = ax.imshow(log_mel(load_16k(item["path"])), origin="lower", aspect="auto",
                       extent=[0, 5, 0, N_MELS], cmap="magma")
        ax.set_ylabel("pasmo mel", fontsize=8, color=INK_MUTED)
        style(ax)
        fig.colorbar(im, ax=ax, pad=0.01).ax.tick_params(labelsize=7, colors=INK_MUTED)

    axes[0, 0].set_title("Sygnał surowy (44,1 kHz, amplituda)", loc="left", fontsize=10, color=INK)
    axes[0, 1].set_title("Po przetwarzaniu: log-mel-spektrogram 16 kHz, 128 × 501 [dB]",
                         loc="left", fontsize=10, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("czas [s]", fontsize=9, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "przyklady_sygnal_spektrogram.png", dpi=150)
    plt.close(fig)


def plot_feature_distributions(df):
    """Wykresy pudełkowe 3 cech w 5 grupach klas."""
    panels = [("rms_db", "Głośność RMS [dBFS]"),
              ("centroid", "Środek ciężkości widma [Hz]\n(„jasność” dźwięku)"),
              ("silence_frac", "Udział ciszy w nagraniu")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURFACE)
    for ax, (col, title) in zip(axes, panels):
        data = [df.loc[df["group"] == g, col] for g in GROUPS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops={"color": INK, "linewidth": 1.5},
                        whiskerprops={"color": INK_MUTED}, capprops={"color": INK_MUTED},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_MUTED,
                                    "markeredgecolor": "none", "alpha": 0.5})
        for patch, color in zip(bp["boxes"], GROUP_COLORS):
            patch.set(facecolor=color, edgecolor=SURFACE, alpha=0.9)
        ax.set_xticks(range(1, 6), [g.replace(" ", "\n", 1) for g in GROUPS], fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax)
    fig.suptitle("Rozkład cech sygnału w grupach klas (2000 nagrań)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rozklad_cech.png", dpi=150)
    plt.close(fig)


def plot_embeddings(df, emb):
    """Siatka 3 metody x 5 grup: w każdym panelu podświetlona jedna grupa, reszta szara."""
    fig, axes = plt.subplots(len(emb), 5, figsize=(16, 3.3 * len(emb)), facecolor=SURFACE)
    for r, (name, xy) in enumerate(emb.items()):
        for c, g in enumerate(GROUPS):
            ax = axes[r, c]
            mask = (df["group"] == g).to_numpy()
            ax.scatter(xy[~mask, 0], xy[~mask, 1], s=4, color=BACKDROP, linewidths=0)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=8, color=GROUP_COLORS[c],
                       edgecolors=SURFACE, linewidths=0.3)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GRID)
            if r == 0:
                ax.set_title(g, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, color=INK)
    fig.suptitle("Przestrzeń cech MFCC (40 cech na nagranie) zrzutowana do 2D",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "przestrzen_cech.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    quality = pd.read_csv(TABLE_DIR / "jakosc_plikow.csv")[["filename", "silence_frac", "longest_zero_run_s"]]
    df = df.merge(quality, on="filename", how="left")

    print("1/4 Przykłady sygnału i spektrogramu...")
    plot_examples(df)

    print("2/4 Cechy wszystkich nagrań...")
    mfcc, df["rms_db"], df["centroid"] = compute_features(df)
    plot_feature_distributions(df)
    print(df.groupby("group")[["rms_db", "centroid", "silence_frac"]].median().round(2)
          .reindex(GROUPS).to_string())

    # Skaler dopasowany tylko do zbioru treningowego (ochrona przed przeciekiem), użyty dla wszystkich
    train_mask = (df["split"] == "trening").to_numpy()
    X = StandardScaler().fit(mfcc[train_mask]).transform(mfcc)

    print("3/4 PCA, t-SNE, UMAP...")
    pca = PCA(n_components=2, random_state=0).fit(X[train_mask])
    emb = {
        "PCA": pca.transform(X),
        "t-SNE": TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X),
        "UMAP": UMAP(n_components=2, n_neighbors=15, random_state=0, n_jobs=1).fit_transform(X),
    }
    plot_embeddings(df, emb)
    print(f"PCA: 2 składowe wyjaśniają {100 * pca.explained_variance_ratio_.sum():.1f}% wariancji")

    print("4/4 Isolation Forest...")
    forest = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(X[train_mask])
    df["anomaly_score"] = -forest.score_samples(X)          # większa wartość = bardziej nietypowe
    df["is_outlier"] = forest.predict(X) == -1
    cols = ["filename", "split", "category", "group", "anomaly_score", "rms_db", "silence_frac", "longest_zero_run_s"]
    top = df.sort_values("anomaly_score", ascending=False)[cols]
    top.head(30).to_csv(TABLE_DIR / "nietypowe_isolation_forest.csv", index=False)
    print(f"Oznaczone jako nietypowe: {df['is_outlier'].sum()} z {len(df)} "
          f"(trening {df.loc[train_mask, 'is_outlier'].sum()})")
    print("Grupy wśród nietypowych:", df.loc[df["is_outlier"], "group"].value_counts().to_dict())
    print("\n10 najbardziej nietypowych nagrań:")
    print(top.head(10).round(2).to_string(index=False))

    print(f"\nZapisano wykresy w {FIG_DIR} i tabelę {TABLE_DIR / 'nietypowe_isolation_forest.csv'}")
