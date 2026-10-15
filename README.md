# esc50-audio-classifier

Rozpoznawanie dźwięków otoczenia (ESC-50, 50 klas) przy użyciu dwóch modeli: CNN 2D na spektrogramie oraz 1D-CNN + GRU na surowym sygnale.

## Uruchomienie

### Środowisko wirtualne

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Instalacja zależności

```bash
pip install -r requirements.txt
```

`requirements.txt` instaluje PyTorch z CUDA 12.8 (wymagane dla kart RTX 50xx).

### Dane

Pobierz zbiór [ESC-50](https://github.com/karolpiczak/ESC-50) i rozpakuj go tak, aby powstał folder `data/ESC-50-master/` (z podfolderami `audio/` i `meta/`). Zawartość `data/` nie jest śledzona przez git.

### Sprawdzenie danych i podziału na zbiory

```bash
python src/dane.py
```

Skrypt wypisuje liczbę nagrań w zbiorach trening/walidacja/test, brakujące pliki audio oraz kontrolę wycieku nagrań źródłowych między zbiorami.
