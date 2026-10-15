"""EDA, krok 1: skan jakości wszystkich plików audio ESC-50.

Dla każdego pliku sprawdza: czy da się go odczytać, częstotliwość próbkowania, liczbę kanałów,
długość, wartości NaN/Inf, poziom głośności, przesterowanie, udział ciszy, najdłuższy ciąg zer
oraz sumę kontrolną MD5 (do wykrycia identycznych duplikatów).
Wynik: docs/eda/jakosc_plikow.csv + podsumowanie w terminalu.
"""
import hashlib

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from dane import ROOT, load_metadata

OUT_DIR = ROOT / "docs" / "eda"

FRAME_S = 0.05          # długość ramki do wykrywania ciszy: 50 ms
SILENCE_DB = -60.0      # ramka cichsza niż -60 dBFS = cisza
CLIP_LEVEL = 0.999      # |próbka| >= 0.999 = przesterowanie


def to_db(value):
    """Amplituda (0..1) -> decybele względem pełnej skali (dBFS)."""
    return 20 * np.log10(value + 1e-12)


def longest_zero_run(x):
    """Długość najdłuższego ciągu próbek równych dokładnie 0."""
    is_zero = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    edges = np.diff(is_zero)
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def analyze_file(path):
    """Zwraca słownik z cechami jakości jednego pliku."""
    try:
        info = sf.info(path)
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as err:                      # plik uszkodzony lub nieczytelny
        return {"read_error": str(err)}

    x = x.mean(axis=1)                            # na wypadek nagrań stereo -> mono
    n_frames = len(x) // int(FRAME_S * sr)
    frames = x[: n_frames * int(FRAME_S * sr)].reshape(n_frames, -1)
    frame_db = to_db(np.sqrt((frames ** 2).mean(axis=1)))

    return {
        "read_error": "",
        "sample_rate": sr,
        "channels": info.channels,
        "subtype": info.subtype,
        "n_samples": len(x),
        "duration_s": len(x) / sr,
        "has_nan_inf": bool(~np.isfinite(x).all()),
        "peak_db": float(to_db(np.abs(x).max())),
        "rms_db": float(to_db(np.sqrt((x ** 2).mean()))),
        "clipped_frac": float((np.abs(x) >= CLIP_LEVEL).mean()),
        "silence_frac": float((frame_db < SILENCE_DB).mean()),
        "longest_zero_run_s": longest_zero_run(x) / sr,
        "md5": hashlib.md5(x.tobytes()).hexdigest(),
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()

    rows = [analyze_file(p) for p in tqdm(df["path"], desc="Skan plików")]
    q = pd.concat([df[["filename", "fold", "target", "category", "src_file"]], pd.DataFrame(rows)], axis=1)
    q.to_csv(OUT_DIR / "jakosc_plikow.csv", index=False)

    ok = q[q["read_error"] == ""]
    print("\n=== Format i integralność ===")
    print(f"Nieczytelne pliki:          {(q['read_error'] != '').sum()}")
    print(f"Częstotliwości próbkowania: {ok['sample_rate'].value_counts().to_dict()}")
    print(f"Liczba kanałów:             {ok['channels'].value_counts().to_dict()}")
    print(f"Format próbek:              {ok['subtype'].value_counts().to_dict()}")
    print(f"Długości [s]:               {ok['duration_s'].round(3).value_counts().to_dict()}")
    print(f"Pliki z NaN/Inf:            {ok['has_nan_inf'].sum()}")

    dup = ok[ok.duplicated("md5", keep=False)].sort_values("md5")
    print(f"Identyczne duplikaty (MD5): {len(dup)} plików")
    if len(dup):
        print(dup[["filename", "category", "md5"]].to_string(index=False))

    print("\n=== Potencjalnie nietypowe próbki ===")
    checks = {
        "przesterowanie > 0,1% próbek": ok["clipped_frac"] > 0.001,
        "cisza > 50% nagrania": ok["silence_frac"] > 0.5,
        "cisza > 90% nagrania": ok["silence_frac"] > 0.9,
        "ciąg zer > 1 s (dopełnienie ciszą)": ok["longest_zero_run_s"] > 1.0,
        "bardzo ciche (szczyt < -30 dBFS)": ok["peak_db"] < -30,
    }
    for name, mask in checks.items():
        print(f"{name:38s} {mask.sum():4d} plików")

    print("\nNajbardziej 'ciche' nagrania (największy udział ciszy):")
    print(ok.nlargest(8, "silence_frac")[["filename", "category", "silence_frac", "longest_zero_run_s", "peak_db"]]
          .round(2).to_string(index=False))

    print("\nKlasy z największym udziałem dopełnienia zerami (średni najdłuższy ciąg zer [s]):")
    print(ok.groupby("category")["longest_zero_run_s"].mean().sort_values(ascending=False).head(8).round(2).to_string())

    print(f"\nZapisano: {OUT_DIR / 'jakosc_plikow.csv'}")
