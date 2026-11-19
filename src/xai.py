"""Zajęcia 10: wyjaśnialność modeli (XAI) na zbiorze testowym (fold 5).

1. Grad-CAM dla CNN: które fragmenty spektrogramu (czas × częstotliwość) zdecydowały o wyniku.
2. Integrated Gradients (Captum) dla GRU: które chwile surowego sygnału zdecydowały o wyniku.
3. Test ciszy: w nagraniach z dużą ilością cyfrowej ciszy zastępujemy ciszę cichym szumem
   i sprawdzamy, czy modele zmieniają decyzje (czy „uczyły się ciszy” zamiast dźwięku).
4. Profil częstotliwości Grad-CAM: na które pasma CNN patrzy średnio (kontrola obciętego pasma).

Użycie:
    python src/xai.py
Wynik:
    docs/figures/xai/gradcam_cnn.png, ig_gru.png, profil_czestotliwosci_cnn.png
    docs/wyniki/xai_wyniki.md
"""
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dane import ROOT, load_metadata
from ewaluacja import load_model
from potok import N_MELS, make_loaders, peak_normalize
from przygotuj_cache import SR

FIG_DIR = ROOT / "docs" / "figures" / "xai"
RESULTS_DIR = ROOT / "docs" / "wyniki"
CNN_CKPT = ROOT / "models" / "cnn_aug_lr0.001_cosine_val4_test5.pt"     # ostateczny CNN (decyzja 2026-11-19)
GRU_CKPT = ROOT / "models" / "gru_aug_lr0.001_cosine_val4_test5.pt"
ERROR_EXAMPLES = ["5-179496-B-16.wav", "5-172639-A-5.wav", "5-117122-A-42.wav"]   # z analizy błędów
HOP, N_FRAMES = 160, 500            # ramki po 10 ms, jak w spektrogramie
SILENCE_DB = -60                    # ramka cichsza niż -60 dBFS = cisza
MIN_SILENCE_SHARE = 0.2             # do testu ciszy: nagrania z co najmniej 20% ciszy
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
IG_STEPS = 32
NOISE_STD = 0.003                   # drugi baseline IG: cichy szum tła (ok. -50 dBFS), 3 losowania


class SpecNet(nn.Module):
    """CNN bez części liczącej spektrogram (wejście = spektrogram [B, 1, 128, 501]), potrzebne dla Grad-CAM."""

    def __init__(self, cnn):
        super().__init__()
        self.blocks, self.pool, self.head = cnn.blocks, cnn.pool, cnn.head

    def forward(self, spec):
        return self.head(self.pool(self.blocks(spec)))


def silence_mask(wave):
    """Dla 500 ramek po 10 ms: True, jeśli ramka jest cicha (< -60 dBFS)."""
    frames = wave[:N_FRAMES * HOP].reshape(N_FRAMES, HOP)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12) < SILENCE_DB


def frame_profile(attr):
    """Atrybucja dla każdej próbki -> suma wartości bezwzględnych w każdej ramce 10 ms."""
    return np.abs(attr[:N_FRAMES * HOP]).reshape(N_FRAMES, HOP).sum(axis=1)


def window_signed(attr, win=1600):
    """Suma atrybucji ze znakiem w oknach 100 ms -> (czas najbardziej dodatniego, czas najbardziej ujemnego okna)."""
    w = attr[:len(attr) // win * win].reshape(-1, win).sum(axis=1)
    return (w.argmax() + 0.5) * win / SR, (w.argmin() + 0.5) * win / SR


def plot_ig_baselines(examples, waves, attrs_zero, attrs_noise, files, labels, probs, names, path):
    t_frames = (np.arange(N_FRAMES) + 0.5) * HOP / SR
    fig, axes = plt.subplots(len(examples), 1, figsize=(13, 1.8 * len(examples)), facecolor="white")
    for ax, i in zip(axes, examples):
        ax.plot(np.arange(waves.shape[-1]) / SR, waves[i] * 0.5, color="#d9d8d4", linewidth=0.4)
        for attr, color, label in [(attrs_zero[i], "#2a78d6", "baseline: cisza (zera)"),
                                   (attrs_noise[i], "#eb6834", "baseline: cichy szum")]:
            prof = np.convolve(frame_profile(attr), np.ones(5) / 5, mode="same")
            ax.plot(t_frames, prof / (prof.max() + 1e-12), color=color, linewidth=1.5, label=label)
        ax.set_xlim(0, 5)
        ax.set_title(describe(i, files, labels, probs, names), loc="left", fontsize=8, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("czas [s]", fontsize=8, color=MUTED)
    fig.suptitle("GRU: Integrated Gradients, porównanie dwóch punktów odniesienia (ważność znormalizowana)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict(fn, x, device, batch=50):
    with torch.no_grad():
        return torch.cat([torch.softmax(fn(x[i:i + batch].to(device)), dim=1).cpu()
                          for i in range(0, len(x), batch)]).numpy()


def pick_examples(files, labels, p_cnn, p_gru, groups):
    """3 trafne przykłady (najpewniejsze, z różnych grup, trafne dla obu modeli) + 3 błędy z analizy błędów."""
    correct = [i for i in np.argsort(-p_cnn.max(axis=1))
               if p_cnn[i].argmax() == labels[i] and p_gru[i].argmax() == labels[i]]
    chosen, used = [], set()
    for i in correct:
        if groups[i] not in used:
            chosen.append(i)
            used.add(groups[i])
        if len(chosen) == 3:
            break
    return chosen + [files.index(f) for f in ERROR_EXAMPLES if f in files]


def describe(i, files, labels, probs, names):
    pred = probs[i].argmax()
    mark = "trafnie" if pred == labels[i] else "BŁĄD"
    return f"{files[i]} | prawda: {names[labels[i]]} | predykcja: {names[pred]} ({probs[i].max():.2f}, {mark})"


def plot_gradcam(examples, specs, cams, files, labels, probs, names, path):
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 2.3 * len(examples)), facecolor="white")
    for row, i in enumerate(examples):
        spec = specs[i]
        axes[row, 0].imshow(spec, origin="lower", aspect="auto", extent=[0, 5, 0, N_MELS], cmap="magma")
        axes[row, 1].imshow(spec, origin="lower", aspect="auto", extent=[0, 5, 0, N_MELS], cmap="gray")
        axes[row, 1].imshow(cams[i], origin="lower", aspect="auto", extent=[0, 5, 0, N_MELS], cmap="jet", alpha=0.45)
        axes[row, 0].set_title(describe(i, files, labels, probs, names), loc="left", fontsize=8, color=INK)
        for ax in axes[row]:
            ax.set_ylabel("pasmo mel", fontsize=7, color=MUTED)
            ax.tick_params(labelsize=7, colors=MUTED)
    axes[0, 1].set_title("Grad-CAM: na co patrzył model (czerwone = ważne)", loc="left", fontsize=8, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("czas [s]", fontsize=8, color=MUTED)
    fig.suptitle("CNN: Grad-CAM na spektrogramie (zbiór testowy)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ig(examples, waves, attrs, files, labels, probs, names, path):
    t_wave = np.arange(waves.shape[-1]) / SR
    t_frames = (np.arange(N_FRAMES) + 0.5) * HOP / SR
    fig, axes = plt.subplots(len(examples), 1, figsize=(13, 1.9 * len(examples)), facecolor="white")
    for ax, i in zip(axes, examples):
        prof = np.convolve(frame_profile(attrs[i]), np.ones(5) / 5, mode="same")     # wygładzenie (50 ms)
        prof = prof / (prof.max() + 1e-12)
        ax.plot(t_wave, waves[i], color="#b8b7b3", linewidth=0.4, label="sygnał")
        ax.fill_between(t_frames, 0, prof, color="#eb6834", alpha=0.6, label="ważność (Integrated Gradients)")
        ax.set_xlim(0, 5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(describe(i, files, labels, probs, names), loc="left", fontsize=8, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("czas [s]", fontsize=8, color=MUTED)
    fig.suptitle("GRU: Integrated Gradients na surowym sygnale (zbiór testowy)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_freq_profile(cams, path):
    freqs = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SR / 2, htk=False)[1:-1]
    profile = cams.mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor="white")
    ax.plot(freqs, profile, color="#2a78d6", linewidth=2)
    ax.set_xlabel("częstotliwość [Hz] (środek pasma mel)", fontsize=9, color=MUTED)
    ax.set_ylabel("średnia ważność Grad-CAM", fontsize=9, color=MUTED)
    ax.set_title("CNN: na które częstotliwości średnio patrzy model (400 nagrań testowych)", loc="left", fontsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return freqs, profile


def _version(pkg):
    from importlib.metadata import version
    try:
        return version(pkg)
    except Exception:
        return "?"


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()
    names = df.drop_duplicates("target").sort_values("target")["category"].tolist()
    _, audio, idx = make_loaders(4, 5, augment=False, num_workers=0)
    rows = idx["test"]
    files = df.loc[rows, "filename"].tolist()
    labels = df.loc[rows, "target"].to_numpy()
    groups = df.loc[rows, "group"].to_numpy()
    waves = torch.stack([peak_normalize(torch.from_numpy(audio[r])) for r in rows]).unsqueeze(1)   # [400, 1, 80000]

    cnn, _ = load_model(CNN_CKPT, device)
    gru, _ = load_model(GRU_CKPT, device)
    specnet = SpecNet(cnn).eval()

    print("1/4 Predykcje...")
    with torch.no_grad():
        specs = torch.cat([cnn.frontend(waves[i:i + 50].to(device)) for i in range(0, len(waves), 50)])
    p_cnn = predict(specnet, specs, device)
    p_gru = predict(gru, waves, device)
    print(f"    accuracy na teście: CNN {np.mean(p_cnn.argmax(1) == labels):.3f}, GRU {np.mean(p_gru.argmax(1) == labels):.3f}")

    print("2/4 Grad-CAM dla wszystkich 400 nagrań...")
    cams = np.zeros((len(waves), N_MELS, specs.shape[-1]), dtype=np.float32)
    with GradCAM(model=specnet, target_layers=[specnet.blocks[3][2]]) as cam:      # ReLU ostatniego bloku (16 x 62)
        for i in range(0, len(waves), 25):
            targets = [ClassifierOutputTarget(int(c)) for c in p_cnn[i:i + 25].argmax(axis=1)]
            cams[i:i + 25] = cam(input_tensor=specs[i:i + 25], targets=targets)

    examples = pick_examples(files, labels, p_cnn, p_gru, groups)
    specs_np = specs[:, 0].cpu().numpy()
    plot_gradcam(examples, specs_np, cams, files, labels, p_cnn, names, FIG_DIR / "gradcam_cnn.png")

    print("3/4 Integrated Gradients dla GRU (przykłady)...")
    ig = IntegratedGradients(gru)
    attrs, attrs_noise, noise_runs, ig_rows = {}, {}, {}, []
    gen_b = torch.Generator().manual_seed(1)
    with torch.backends.cudnn.flags(enabled=False):     # GRU w cuDNN nie liczy gradientów w trybie eval
        for i in examples:
            x = waves[i:i + 1].to(device)
            target = int(p_gru[i].argmax())               # wyjaśniamy klasę PRZEWIDZIANĄ przez model
            a = ig.attribute(x, baselines=torch.zeros_like(x), target=target,
                             n_steps=IG_STEPS, internal_batch_size=8)
            attrs[i] = a[0, 0].detach().cpu().numpy()
            runs = []
            for _ in range(3):                           # baseline = cichy szum, 3 losowania
                base = (torch.randn(x.shape, generator=gen_b) * NOISE_STD).to(device)
                runs.append(ig.attribute(x, baselines=base, target=target, n_steps=IG_STEPS,
                                         internal_batch_size=8)[0, 0].detach().cpu().numpy())
            noise_runs[i] = runs
            attrs_noise[i] = np.mean(runs, axis=0)
            pz, pn = frame_profile(attrs[i]), frame_profile(attrs_noise[i])
            stab = np.mean([np.corrcoef(frame_profile(runs[a_]), frame_profile(runs[b_]))[0, 1]
                            for a_, b_ in [(0, 1), (0, 2), (1, 2)]])
            sil_i = silence_mask(waves[i, 0].numpy())
            pos, neg = window_signed(attrs_noise[i])
            ig_rows.append((i, np.corrcoef(pz, pn)[0, 1], stab, sil_i.mean(),
                            (pn * sil_i).sum() / (pn.sum() + 1e-12), pos, neg))
    plot_ig(examples, waves[:, 0].numpy(), attrs, files, labels, p_gru, names, FIG_DIR / "ig_gru.png")
    plot_ig_baselines(examples, waves[:, 0].numpy(), attrs, attrs_noise, files, labels, p_gru, names,
                      FIG_DIR / "ig_baseline_porownanie.png")

    print("4/4 Test ciszy i profil częstotliwości...")
    sil = np.stack([silence_mask(w) for w in waves[:, 0].numpy()])              # [400, 500]
    sel = np.where(sil.mean(axis=1) >= MIN_SILENCE_SHARE)[0]
    noisy = waves.clone()
    gen = torch.Generator().manual_seed(0)
    for i in sel:
        sample_mask = torch.from_numpy(np.repeat(sil[i], HOP))
        noise = torch.randn(int(sample_mask.sum()), generator=gen) * 0.003      # ok. -50 dBFS
        noisy[i, 0, :N_FRAMES * HOP][sample_mask] = noise
    with torch.no_grad():
        specs_noisy = torch.cat([cnn.frontend(noisy[i:i + 50].to(device)) for i in range(0, len(noisy), 50)])
    q_cnn = predict(specnet, specs_noisy, device)
    q_gru = predict(gru, noisy, device)

    cnn_prof = cams[:, :, :N_FRAMES].sum(axis=1)                                  # ważność CNN w czasie
    cam_in_silence = np.mean([(cnn_prof[i] * sil[i]).sum() / (cnn_prof[i].sum() + 1e-12) for i in sel])
    freqs, fprof = plot_freq_profile(cams, FIG_DIR / "profil_czestotliwosci_cnn.png")
    high_share = fprof[freqs >= 4000].sum() / fprof.sum()

    def acc(p, s):
        return np.mean(p[s].argmax(1) == labels[s])

    lines = [
        "# Wyniki XAI (zbiór testowy, fold 5)", "",
        "## Test ciszy: cisza zastąpiona cichym szumem (ok. -50 dBFS)", "",
        f"Nagrania z co najmniej {int(MIN_SILENCE_SHARE * 100)}% ciszy: **{len(sel)}** z {len(waves)} "
        f"(średnio {100 * sil[sel].mean():.0f}% ciszy w tych nagraniach).", "",
        "| Model | Accuracy: oryginał | Accuracy: cisza → szum | Zmienione predykcje |",
        "|---|---|---|---|",
        f"| CNN | {acc(p_cnn, sel):.3f} | {acc(q_cnn, sel):.3f} | {np.mean(p_cnn[sel].argmax(1) != q_cnn[sel].argmax(1)):.1%} |",
        f"| GRU | {acc(p_gru, sel):.3f} | {acc(q_gru, sel):.3f} | {np.mean(p_gru[sel].argmax(1) != q_gru[sel].argmax(1)):.1%} |",
        "",
        f"Udział ważności Grad-CAM (CNN) przypadający na ciszę: **{cam_in_silence:.1%}**, "
        f"przy udziale ciszy w czasie {sil[sel].mean():.1%}.", "",
        "## Profil częstotliwości Grad-CAM (CNN)", "",
        f"Udział ważności w pasmach powyżej 4 kHz: **{high_share:.1%}** "
        f"(pasma powyżej 4 kHz to {np.mean(freqs >= 4000):.0%} wszystkich pasm mel).", "",
        "## Integrated Gradients (GRU): porównanie baseline'ów i stabilność", "",
        "Baseline 1: cisza (same zera). Baseline 2: cichy szum (std 0,003, ok. -50 dBFS), średnia z 3 losowań.",
        "Korelacja = zgodność profili ważności w czasie (ramki 10 ms); 1 = identyczne.", "",
        "| Plik | Korelacja: zera vs szum | Stabilność: korelacja między losowaniami szumu | Udział ciszy w czasie | "
        "Udział ważności (baseline szum) w ciszy | Najbardziej dodatnie okno 100 ms [s] | Najbardziej ujemne okno [s] |",
        "|---|---|---|---|---|---|---|",
    ] + [f"| {files[i]} | {c:.2f} | {st:.2f} | {sh:.0%} | {ins:.0%} | {pos:.2f} | {neg:.2f} |"
         for i, c, st, sh, ins, pos, neg in ig_rows] + [
        "", "## Parametry eksperymentu", "",
        f"- Modele: `{CNN_CKPT.name}`, `{GRU_CKPT.name}`; tryb `eval()`; ten sam preprocessing co w treningu "
        "(16 kHz, normalizacja amplitudy, bez augmentacji).",
        "- Wyjaśniana klasa: przewidziana przez model (logit przed softmaxem).",
        "- Grad-CAM: warstwa `blocks[3][2]` (ReLU ostatniego bloku, mapa 16 × 62), mapa skalowana do 128 × 501 "
        "i normalizowana do 0–1 osobno dla każdego nagrania.",
        f"- Integrated Gradients: {IG_STEPS} kroki, baseline zerowy i szumowy (std {NOISE_STD}), cuDNN wyłączony na czas liczenia.",
        f"- Test ciszy: ramki 10 ms ciszej niż {SILENCE_DB} dBFS, nagrania z ≥ {int(MIN_SILENCE_SHARE * 100)}% ciszy, "
        "szum std 0,003, ziarno 0.",
        "- Wersje: " + ", ".join(f"{pkg} {_version(pkg)}" for pkg in
                                 ["torch", "torchaudio", "captum", "grad-cam", "numpy", "librosa"]) + ".",
        "", "## Przykłady na wykresach", "",
    ] + [f"- {describe(i, files, labels, p_cnn, names)} | GRU: {names[p_gru[i].argmax()]} ({p_gru[i].max():.2f})"
         for i in examples]
    out = RESULTS_DIR / "xai_wyniki.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nZapisano: {out}\n          {FIG_DIR}")
