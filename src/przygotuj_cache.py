"""Potok danych, krok 1: jednorazowe przygotowanie danych 16 kHz (cache).

Wczytuje wszystkie 2000 nagrań (44,1 kHz), przepróbkowuje je do 16 kHz i zapisuje jako jedną
tablicę [2000, 80000] w data/cache/audio_16k.npy (kolejność wierszy = kolejność w esc50.csv).
Z tego pliku korzystają oba modele, więc trening nie musi co epokę dekodować plików WAV.
Uruchamiamy raz (ok. 1 min); ponowne uruchomienie nadpisuje cache.
"""
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm

from dane import ROOT, load_metadata

CACHE_DIR = ROOT / "data" / "cache"
AUDIO_CACHE = CACHE_DIR / "audio_16k.npy"
FILES_CACHE = CACHE_DIR / "audio_16k_pliki.txt"

SR_IN, SR = 44100, 16000
N_SAMPLES = 5 * SR                     # 80 000 próbek = 5 s


def to_16k(x, sr_in=SR_IN):
    """Przepróbkowanie sygnału mono do 16 kHz i dopasowanie długości do dokładnie 80 000 próbek.
    Tej samej funkcji użyjemy w aplikacji dla nagrań użytkownika."""
    x = torch.as_tensor(x, dtype=torch.float32)
    if sr_in != SR:
        x = AF.resample(x, orig_freq=sr_in, new_freq=SR)
    if len(x) < N_SAMPLES:                                   # za krótkie: dopełnij zerami
        x = torch.nn.functional.pad(x, (0, N_SAMPLES - len(x)))
    return x[:N_SAMPLES]                                     # za długie: przytnij


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()

    audio = np.zeros((len(df), N_SAMPLES), dtype=np.float32)
    for i, path in enumerate(tqdm(df["path"], desc="Przepróbkowanie do 16 kHz")):
        x, sr = sf.read(path, dtype="float32")
        audio[i] = to_16k(x, sr).numpy()

    np.save(AUDIO_CACHE, audio)
    FILES_CACHE.write_text("\n".join(df["filename"]))

    print(f"Kształt: {audio.shape}, typ: {audio.dtype}, rozmiar: {audio.nbytes / 1e6:.0f} MB")
    print(f"Zakres wartości: [{audio.min():.3f}, {audio.max():.3f}], NaN/Inf: {int((~np.isfinite(audio)).sum())}")
    print(f"Zapisano: {AUDIO_CACHE}")
