"""Wczytanie metadanych ESC-50 i podział na zbiory według oficjalnych foldów."""
from pathlib import Path

import pandas as pd

# Ścieżki liczone względem tego pliku, więc działają niezależnie od tego, skąd uruchamiamy skrypt
ROOT = Path(__file__).resolve().parent.parent          # główny folder repozytorium
DATA_DIR = ROOT / "data" / "ESC-50-master"
AUDIO_DIR = DATA_DIR / "audio"
META_CSV = DATA_DIR / "meta" / "esc50.csv"

# Podział ustalony w planie: foldy 1-3 trening, 4 walidacja, 5 test
TRAIN_FOLDS = [1, 2, 3]
VAL_FOLDS = [4]
TEST_FOLDS = [5]


def load_metadata():
    """Wczytuje esc50.csv i dodaje kolumnę z pełną ścieżką do pliku audio."""
    df = pd.read_csv(META_CSV)
    df["path"] = df["filename"].apply(lambda name: AUDIO_DIR / name)
    return df


def split_by_folds(df):
    """Dzieli tabelę na trening, walidację i test według numeru foldu."""
    train = df[df["fold"].isin(TRAIN_FOLDS)].reset_index(drop=True)
    val = df[df["fold"].isin(VAL_FOLDS)].reset_index(drop=True)
    test = df[df["fold"].isin(TEST_FOLDS)].reset_index(drop=True)
    return train, val, test


if __name__ == "__main__":
    df = load_metadata()
    train, val, test = split_by_folds(df)

    print(f"Wszystkich nagrań: {len(df)}, klas: {df['target'].nunique()}")
    print(f"Trening: {len(train)} | Walidacja: {len(val)} | Test: {len(test)}")

    missing = [p for p in df["path"] if not p.exists()]
    print(f"Brakujących plików audio: {len(missing)}")

    # Kontrola wycieku: liczba nagrań źródłowych (src_file) wspólnych dla dwóch zbiorów.
    # Kluczowe są pary z treningiem. 2 wspólne nagrania walidacja-test są znane i opisane
    # w rejestrze decyzji (różne klasy z jednego nagrania, np. kura/kogut).
    tr, va, te = (set(s["src_file"]) for s in (train, val, test))
    print(f"Wspólne nagrania źródłowe: trening-walidacja {len(tr & va)}, "
          f"trening-test {len(tr & te)}, walidacja-test {len(va & te)}")