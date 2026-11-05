# Zestawienie treningów (walidacja)

| Eksperyment | Model | Augmentacja | Parametry | Najlepsza epoka | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Czas epoki [s] | Łącznie [min] |
|---|---|---|---|---|---|---|---|---|---|---|
| gru_aug_val4_test5 | GRU | tak | 382 258 | 67 / 87 | 0.598 | 0.576 | 1.409 | 0.914 | 1.1 | 1.9 |
| gru_aug_lr0.0003_cosine_val4_test5 | GRU | tak | 382 258 | 96 / 100 | 0.535 | 0.509 | 1.601 | 0.746 | 1.1 | 2.2 |
| gru_aug_lr0.0003_plateau_val4_test5 | GRU | tak | 382 258 | 85 / 100 | 0.550 | 0.528 | 1.551 | 0.798 | 1.1 | 2.2 |
