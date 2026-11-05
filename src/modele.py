"""Architektury obu modeli (zaprojektowane przez nas, uczone od zera) + kontrola przed treningiem.

Oba modele przyjmują ten sam batch sygnałów [B, 1, 80000] z potok.py i zwracają logity [B, 50],
więc jedna pętla treningowa obsłuży oba.
- CNNModel: sygnał -> LogMel (na GPU) -> [trening] SpecAugment -> 4 bloki Conv2d -> GAP -> Linear
- GRUModel: sygnał -> 3 bloki Conv1d (skracają 80 000 próbek do 250 kroków) -> dwukierunkowy GRU -> średnia -> Linear
Inicjalizacja: losowa, domyślna w PyTorch (Kaiming-uniform dla warstw Conv i Linear).
"""
import torch
import torch.nn as nn

from potok import LogMel, SpecAugment

N_CLASSES = 50


# ---------- Model 1: CNN 2D na log-mel-spektrogramie ----------

def conv_block_2d(c_in, c_out):
    """Conv 3x3 (wykrywa lokalne wzorce) -> BatchNorm (stabilizuje) -> ReLU -> MaxPool 2x2 (zmniejsza 2x)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNNModel(nn.Module):
    def __init__(self, mel_mean=None, mel_std=None, dropout=0.3, use_specaug=True):
        super().__init__()
        self.frontend = LogMel(mel_mean, mel_std)      # bez parametrów trenowalnych
        self.specaug = SpecAugment() if use_specaug else nn.Identity()   # działa tylko w trybie treningu
        self.blocks = nn.Sequential(
            conv_block_2d(1, 32),
            conv_block_2d(32, 64),
            conv_block_2d(64, 128),
            conv_block_2d(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)            # Global Average Pooling: [B, 256, 8, 31] -> [B, 256, 1, 1]
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, N_CLASSES))

    def forward(self, wave):
        spec = self.frontend(wave)                     # [B, 1, 128, 501]
        if self.training:
            spec = self.specaug(spec)
        return self.head(self.pool(self.blocks(spec)))


# ---------- Model 2: 1D-CNN + GRU na surowym sygnale ----------

def conv_block_1d(c_in, c_out, kernel, stride=1):
    """Conv1d -> BatchNorm -> ReLU -> MaxPool 4 (skraca sekwencję 4x)."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=kernel // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(4),
    )


class GRUModel(nn.Module):
    def __init__(self, hidden=96, gru_layers=2, dropout=0.3):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_block_1d(1, 64, kernel=11, stride=5),   # 80 000 -> 16 000 -> pool -> 4000
            conv_block_1d(64, 128, kernel=3),            # 4000 -> 1000
            conv_block_1d(128, 128, kernel=3),           # 1000 -> 250 kroków po 20 ms
        )
        self.gru = nn.GRU(input_size=128, hidden_size=hidden, num_layers=gru_layers,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, N_CLASSES))

    def forward(self, wave):
        x = self.blocks(wave)              # [B, 128, 250]
        x = x.transpose(1, 2)              # [B, 250, 128]: sekwencja 250 kroków po 128 cech
        x, _ = self.gru(x)                 # [B, 250, 192]: GRU czyta kroki w obie strony
        x = x.mean(dim=1)                  # [B, 192]: średnia po czasie
        return self.head(x)                # [B, 50]


# ---------- Kontrola przed treningiem ----------

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def layer_shapes(model, wave, names):
    """Kształt wyjścia wybranych modułów dla danego wejścia (przez tzw. hooki)."""
    shapes, modules = [], dict(model.named_modules())
    hooks = [modules[n].register_forward_hook(
        lambda m, i, o, n=n: shapes.append((n, tuple((o[0] if isinstance(o, tuple) else o).shape))))
        for n in names]
    with torch.no_grad():
        model(wave)
    for h in hooks:
        h.remove()
    return shapes


if __name__ == "__main__":
    from potok import make_loaders, stats_path

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, _, _ = make_loaders()
    wave, y = next(iter(loaders["val"]))               # prawdziwy batch, bez augmentacji
    wave, y = wave.to(device), y.to(device)
    stats = torch.load(stats_path())

    models = {
        "CNN": (CNNModel(stats["mean"], stats["std"]),
                ["frontend", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "pool", "head"]),
        "GRU": (GRUModel(), ["blocks.0", "blocks.1", "blocks.2", "gru", "head"]),
    }
    loss_fn = nn.CrossEntropyLoss()

    for name, (model, names) in models.items():
        model = model.to(device).eval()
        total, trainable = count_params(model)
        print(f"\n===== {name} =====")
        print(f"Parametry: wszystkie {total:,}, trenowalne {trainable:,}".replace(",", " "))
        print(f"Wejście: {tuple(wave.shape)}")
        for layer, shape in layer_shapes(model, wave, names):
            print(f"  {layer:10s} -> {shape}")

        with torch.no_grad():
            out = model(wave)
            loss = loss_fn(out, y)
        print(f"Wyjście: {tuple(out.shape)} | NaN/Inf: {int((~torch.isfinite(out)).sum())}")
        print(f"Strata na starcie: {loss.item():.3f} (oczekiwane ok. ln(50) = 3.912 dla losowych wag)")

        # Jeden krok treningowy (w przód + wstecz), żeby zmierzyć szczytowe zużycie pamięci GPU
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            model.train()
            loss_fn(model(wave), y).backward()
            print(f"Szczytowa pamięć GPU (batch 32, trening): {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
