# Wyniki XAI (zbiór testowy, fold 5)

## Test ciszy: cisza zastąpiona cichym szumem (ok. -50 dBFS)

Nagrania z co najmniej 20% ciszy: **86** z 400 (średnio 58% ciszy w tych nagraniach).

| Model | Accuracy: oryginał | Accuracy: cisza → szum | Zmienione predykcje |
|---|---|---|---|
| CNN | 0.616 | 0.581 | 10.5% |
| GRU | 0.686 | 0.674 | 3.5% |

Udział ważności Grad-CAM (CNN) przypadający na ciszę: **43.0%**, przy udziale ciszy w czasie 58.4%.

## Profil częstotliwości Grad-CAM (CNN)

Udział ważności w pasmach powyżej 4 kHz: **19.9%** (pasma powyżej 4 kHz to 22% wszystkich pasm mel).

## Integrated Gradients (GRU): porównanie baseline'ów i stabilność

Baseline 1: cisza (same zera). Baseline 2: cichy szum (std 0,003, ok. -50 dBFS), średnia z 3 losowań.
Korelacja = zgodność profili ważności w czasie (ramki 10 ms); 1 = identyczne.

| Plik | Korelacja: zera vs szum | Stabilność: korelacja między losowaniami szumu | Udział ciszy w czasie | Udział ważności (baseline szum) w ciszy | Najbardziej dodatnie okno 100 ms [s] | Najbardziej ujemne okno [s] |
|---|---|---|---|---|---|---|
| 5-223176-A-37.wav | 1.00 | 0.99 | 29% | 0% | 2.75 | 3.85 |
| 5-147297-A-27.wav | 1.00 | 1.00 | 0% | 0% | 2.05 | 4.95 |
| 5-253094-A-49.wav | 1.00 | 0.99 | 0% | 0% | 4.25 | 0.65 |
| 5-179496-B-16.wav | 0.90 | 0.99 | 0% | 0% | 0.95 | 0.55 |
| 5-172639-A-5.wav | 1.00 | 1.00 | 21% | 0% | 2.05 | 4.15 |
| 5-117122-A-42.wav | 0.98 | 0.98 | 0% | 0% | 3.25 | 4.25 |

## Parametry eksperymentu

- Modele: `cnn_aug_lr0.001_cosine_val4_test5.pt`, `gru_aug_lr0.001_cosine_val4_test5.pt`; tryb `eval()`; ten sam preprocessing co w treningu (16 kHz, normalizacja amplitudy, bez augmentacji).
- Wyjaśniana klasa: przewidziana przez model (logit przed softmaxem).
- Grad-CAM: warstwa `blocks[3][2]` (ReLU ostatniego bloku, mapa 16 × 62), mapa skalowana do 128 × 501 i normalizowana do 0–1 osobno dla każdego nagrania.
- Integrated Gradients: 32 kroki, baseline zerowy i szumowy (std 0.003), cuDNN wyłączony na czas liczenia.
- Test ciszy: ramki 10 ms ciszej niż -60 dBFS, nagrania z ≥ 20% ciszy, szum std 0,003, ziarno 0.
- Wersje: torch 2.11.0+cu128, torchaudio 2.11.0+cu128, captum 0.9.0, grad-cam 1.5.7, numpy 2.5.3, librosa 1.0.0.

## Przykłady na wykresach

- 5-223176-A-37.wav | prawda: clock_alarm | predykcja: clock_alarm (1.00, trafnie) | GRU: clock_alarm (1.00)
- 5-147297-A-27.wav | prawda: brushing_teeth | predykcja: brushing_teeth (1.00, trafnie) | GRU: brushing_teeth (0.99)
- 5-253094-A-49.wav | prawda: hand_saw | predykcja: hand_saw (1.00, trafnie) | GRU: hand_saw (1.00)
- 5-179496-B-16.wav | prawda: wind | predykcja: siren (0.99, BŁĄD) | GRU: siren (0.94)
- 5-172639-A-5.wav | prawda: cat | predykcja: crying_baby (0.97, BŁĄD) | GRU: crying_baby (0.83)
- 5-117122-A-42.wav | prawda: siren | predykcja: cow (0.71, BŁĄD) | GRU: cow (0.67)
