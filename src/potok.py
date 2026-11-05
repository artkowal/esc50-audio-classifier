"""Potok danych (zajęcia 3): Dataset i DataLoader dla obu modeli.

Wspólna ścieżka dla obu modeli:
    cache 16 kHz -> normalizacja amplitudy (każde nagranie osobno) -> [tylko trening] augmentacja sygnału
    -> batch sygnałów [B, 1, 80000]
Model GRU dostaje ten batch bezpośrednio.
Model CNN przepuszcza go dodatkowo (na GPU) przez:
    LogMel (log-mel-spektrogram + normalizacja statystykami z treningu) -> [tylko trening] SpecAugment
    -> batch spektrogramów [B, 1, 128, 501]
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from dane import ROOT, load_metadata
from przygotuj_cache import AUDIO_CACHE, FILES_CACHE, N_SAMPLES, SR

N_FFT, HOP, N_MELS = 1024, 160, 128         # parametry spektrogramu ze specyfikacji
BATCH_SIZE = 32
FIG_DIR = ROOT / "docs" / "figures" / "potok"


# ---------- Sygnał: normalizacja i augmentacja ----------

def peak_normalize(x, eps=1e-8):
    """Normalizacja amplitudy jednego nagrania: największa wartość bezwzględna = 1.
    Korzysta tylko z danego nagrania, więc nie ma przecieku danych."""
    return x / (x.abs().max() + eps)


def augment_wave(x):
    """Losowe zmiany sygnału [80000], tylko dla zbioru treningowego (wnioski z EDA)."""
    # 1) przesunięcie w czasie: zdarzenie może być w dowolnym miejscu nagrania
    x = torch.roll(x, shifts=int(torch.randint(0, N_SAMPLES, (1,))))
    # 2) zmiana głośności o -6..+6 dB: model nie powinien opierać się na głośności
    x = x * 10 ** (torch.empty(1).uniform_(-6, 6) / 20)
    # 3) w połowie przypadków szum tła (SNR 20-40 dB): zaciera też „idealną” cyfrową ciszę
    if torch.rand(1) < 0.5:
        snr_db = torch.empty(1).uniform_(20, 40)
        rms = x.pow(2).mean().sqrt()
        x = x + torch.randn_like(x) * rms / 10 ** (snr_db / 20)
    return x


class AudioDataset(Dataset):
    """Zwraca parę (sygnał [1, 80000], etykieta 0-49) dla nagrań o podanych numerach wierszy."""

    def __init__(self, audio, indices, labels, train=False):
        self.audio = audio          # wspólny cache [2000, 80000] (numpy, w RAM)
        self.indices = indices      # które wiersze należą do tego zbioru
        self.labels = labels        # etykiety wszystkich 2000 nagrań
        self.train = train          # augmentacja tylko w treningu

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.indices[i]
        x = peak_normalize(torch.from_numpy(self.audio[row]))
        if self.train:
            x = augment_wave(x)
        return x.unsqueeze(0), int(self.labels[row])       # [1, 80000] (oś kanału), etykieta


def make_loaders(val_fold=4, test_fold=5, batch_size=BATCH_SIZE, augment=True, num_workers=4):
    """Tworzy DataLoadery train/val/test. Domyślnie: trening = foldy 1-3, walidacja = 4, test = 5.
    val_fold/test_fold przydadzą się w walidacji krzyżowej, augment=False w eksperymencie bez augmentacji.
    num_workers: liczba procesów przygotowujących batche równolegle z treningiem na GPU."""
    df = load_metadata()
    assert FILES_CACHE.read_text().split("\n") == df["filename"].tolist(), "Cache nie pasuje do esc50.csv"
    audio = np.load(AUDIO_CACHE)
    labels = df["target"].to_numpy()
    folds = df["fold"].to_numpy()

    idx = {
        "train": np.where((folds != val_fold) & (folds != test_fold))[0],
        "val": np.where(folds == val_fold)[0],
        "test": np.where(folds == test_fold)[0],
    }
    loaders = {
        name: DataLoader(
            AudioDataset(audio, rows, labels, train=(name == "train" and augment)),
            batch_size=batch_size,
            shuffle=(name == "train"),          # losowa kolejność tylko w treningu
            drop_last=(name == "train"),        # bez niepełnego ostatniego batcha w treningu
            pin_memory=torch.cuda.is_available(),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        for name, rows in idx.items()
    }
    return loaders, audio, idx


# ---------- Spektrogram dla CNN (na GPU) ----------

class LogMel(nn.Module):
    """Sygnał [B, 1, 80000] -> znormalizowany log-mel-spektrogram [B, 1, 128, 501]."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                    mel_scale="slaney", norm="slaney", power=2.0)
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80)
        # średnia i odchylenie każdego z 128 pasm, policzone na zbiorze treningowym
        self.register_buffer("mean", torch.zeros(N_MELS, 1) if mean is None else mean)
        self.register_buffer("std", torch.ones(N_MELS, 1) if std is None else std)

    def forward(self, wave):
        spec = self.to_db(self.mel(wave))
        return (spec - self.mean) / self.std


class SpecAugment(nn.Module):
    """Tylko trening: zasłania losowe pasy częstotliwości i fragmenty czasu (osobno w każdym nagraniu)."""

    def __init__(self, freq_mask=16, time_mask=50, n_masks=2):
        super().__init__()
        layers = []
        for _ in range(n_masks):
            layers += [T.FrequencyMasking(freq_mask, iid_masks=True), T.TimeMasking(time_mask, iid_masks=True)]
        self.masks = nn.Sequential(*layers)

    def forward(self, spec):
        return self.masks(spec)


@torch.no_grad()
def compute_mel_stats(audio, rows, device, chunk=64):
    """Średnia i odchylenie każdego pasma mel, liczone TYLKO na nagraniach treningowych, bez augmentacji."""
    logmel = LogMel().to(device)
    total = torch.zeros(N_MELS, dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    count = 0
    for start in range(0, len(rows), chunk):
        wave = torch.from_numpy(audio[rows[start:start + chunk]]).to(device)
        wave = wave / (wave.abs().amax(dim=1, keepdim=True) + 1e-8)      # ta sama normalizacja co w Dataset
        spec = logmel(wave.unsqueeze(1)).double()                          # [b, 1, 128, 501]
        total += spec.sum(dim=(0, 1, 3))
        total_sq += spec.pow(2).sum(dim=(0, 1, 3))
        count += spec.shape[0] * spec.shape[3]
    mean = total / count
    std = (total_sq / count - mean ** 2).clamp_min(1e-8).sqrt()
    return mean.float().unsqueeze(1).cpu(), std.float().unsqueeze(1).cpu()


def stats_path(val_fold=4, test_fold=5):
    return ROOT / "data" / "cache" / f"mel_stats_val{val_fold}_test{test_fold}.pt"


# ---------- Test potoku ----------

def plot_augmentation(audio, row, logmel, specaug, device, path):
    """Jedno nagranie treningowe: bez augmentacji, po augmentacji sygnału, po SpecAugment."""
    clean = peak_normalize(torch.from_numpy(audio[row])).view(1, 1, -1).to(device)
    torch.manual_seed(1)
    augmented = augment_wave(clean.view(-1).cpu()).view(1, 1, -1).to(device)
    with torch.no_grad():
        panels = [logmel(clean), logmel(augmented), specaug(logmel(augmented))]
    titles = ["Bez augmentacji", "Po augmentacji sygnału\n(przesunięcie, głośność, szum)", "Po SpecAugment"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), facecolor="white")
    for ax, spec, title in zip(axes, panels, titles):
        ax.imshow(spec[0, 0].cpu().numpy(), origin="lower", aspect="auto", extent=[0, 5, 0, N_MELS],
                  cmap="magma", vmin=-3, vmax=3)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("czas [s]", fontsize=9)
    axes[0].set_ylabel("pasmo mel", fontsize=9)
    fig.suptitle("Wejście CNN po normalizacji (wartości w odchyleniach standardowych)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Urządzenie: {device}")

    loaders, audio, idx = make_loaders()
    for name, dl in loaders.items():
        print(f"{name:9s}: {len(dl.dataset):4d} nagrań, {len(dl):3d} batchy")

    # Batch dla modelu GRU
    wave, y = next(iter(loaders["train"]))
    print(f"\nBatch GRU (sygnał):     {tuple(wave.shape)} {wave.dtype} | etykiety: {tuple(y.shape)} {y.dtype}")

    # Statystyki normalizacji (tylko trening) i batch dla modelu CNN
    mean, std = compute_mel_stats(audio, idx["train"], device)
    torch.save({"mean": mean, "std": std}, stats_path())
    logmel = LogMel(mean, std).to(device)
    specaug = SpecAugment().to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        spec = specaug(logmel(wave.to(device)))
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"Batch CNN (spektrogram): {tuple(spec.shape)} {spec.dtype} | "
          f"czas na GPU: {1000 * (time.perf_counter() - t0):.1f} ms")

    # Kontrola na walidacji (bez augmentacji): po normalizacji średnia ok. 0, odchylenie ok. 1
    wave_val, _ = next(iter(loaders["val"]))
    with torch.no_grad():
        spec_val = logmel(wave_val.to(device))
    print(f"Walidacja po normalizacji: średnia {spec_val.mean():.2f}, odchylenie {spec_val.std():.2f}")
    print(f"NaN/Inf: sygnał {int((~torch.isfinite(wave)).sum())}, spektrogram {int((~torch.isfinite(spec)).sum())}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_augmentation(audio, idx["train"][0], logmel, specaug, device, FIG_DIR / "augmentacja.png")
    print(f"\nZapisano: {stats_path()}")
    print(f"Zapisano: {FIG_DIR / 'augmentacja.png'}")
