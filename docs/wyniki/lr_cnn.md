# Zestawienie treningów (walidacja)

| Eksperyment | Model | Augmentacja | Parametry | Najlepsza epoka | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Czas epoki [s] | Łącznie [min] |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn_aug_val4_test5 | CNN | tak | 401 650 | 90 / 100 | 0.767 | 0.764 | 0.861 | 0.808 | 2.7 | 5.1 |
| cnn_aug_lr0.0003_cosine_val4_test5 | CNN | tak | 401 650 | 96 / 100 | 0.667 | 0.646 | 1.204 | 0.665 | 2.7 | 5.1 |
| cnn_aug_lr0.0003_plateau_val4_test5 | CNN | tak | 401 650 | 98 / 100 | 0.703 | 0.684 | 1.105 | 0.710 | 2.7 | 5.1 |
