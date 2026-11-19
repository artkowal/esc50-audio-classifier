"""Test odporności na ciszę na wybranym foldzie (domyślnie fold 4 = walidacja, żeby decyzja nie korzystała z testu).

W nagraniach z co najmniej 20% cyfrowej ciszy zastępujemy ciszę cichym szumem (ok. -50 dBFS)
i porównujemy accuracy oraz odsetek zmienionych predykcji. Duży spadek = model opiera się na ciszy
(albo nie jest odporny na szum tła, który w aplikacji z mikrofonem będzie zawsze).

Użycie:
    python src/test_ciszy.py models/cnn_bezaug_lr0.001_cosine_val4_test5.pt models/cnn_aug_lr0.001_cosine_val4_test5.pt models/gru_aug_lr0.001_cosine_val4_test5.pt
Wynik: tabela w terminalu i w docs/wyniki/test_ciszy_fold<f>.md
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dane import ROOT, load_metadata
from ewaluacja import load_model
from potok import peak_normalize
from przygotuj_cache import AUDIO_CACHE
from xai import HOP, MIN_SILENCE_SHARE, N_FRAMES, silence_mask


def predict(model, waves, device, batch=50):
    with torch.no_grad():
        return torch.cat([model(waves[i:i + batch].to(device)).argmax(dim=1).cpu()
                          for i in range(0, len(waves), batch)]).numpy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpointy", nargs="+", type=Path)
    p.add_argument("--fold", type=int, default=4, help="fold do testu (domyślnie 4 = walidacja)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_metadata()
    audio = np.load(AUDIO_CACHE)
    rows = np.where(df["fold"].to_numpy() == args.fold)[0]
    labels = df.loc[rows, "target"].to_numpy()
    waves = torch.stack([peak_normalize(torch.from_numpy(audio[r])) for r in rows]).unsqueeze(1)

    sil = np.stack([silence_mask(w) for w in waves[:, 0].numpy()])
    sel = np.where(sil.mean(axis=1) >= MIN_SILENCE_SHARE)[0]
    noisy = waves.clone()
    gen = torch.Generator().manual_seed(0)
    for i in sel:
        mask = torch.from_numpy(np.repeat(sil[i], HOP))
        noisy[i, 0, :N_FRAMES * HOP][mask] = torch.randn(int(mask.sum()), generator=gen) * 0.003

    lines = [f"# Test ciszy na foldzie {args.fold}", "",
             f"Nagrania z co najmniej {int(MIN_SILENCE_SHARE * 100)}% ciszy: {len(sel)} z {len(rows)} "
             f"(średnio {100 * sil[sel].mean():.0f}% ciszy). Cisza zastąpiona szumem ok. -50 dBFS.", "",
             "| Model | Accuracy (cały fold) | Accuracy: nagrania z ciszą | Po zamianie cisza → szum | Zmienione predykcje |",
             "|---|---|---|---|---|"]
    for path in args.checkpointy:
        model, _ = load_model(path, device)
        orig = predict(model, waves, device)
        mod = predict(model, noisy, device)
        lines.append(f"| {path.stem} | {np.mean(orig == labels):.3f} | {np.mean(orig[sel] == labels[sel]):.3f} | "
                     f"{np.mean(mod[sel] == labels[sel]):.3f} | {np.mean(orig[sel] != mod[sel]):.1%} |")

    out = ROOT / "docs" / "wyniki" / f"test_ciszy_fold{args.fold}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nZapisano: {out}")
