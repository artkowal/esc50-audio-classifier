# Zestawienie treningów (walidacja)

| Eksperyment | Model | Augmentacja | Parametry | Najlepsza epoka | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Czas epoki [s] | Łącznie [min] |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn_aug_val4_test5 | CNN | tak | 401 650 | 90 / 100 | 0.767 | 0.764 | 0.861 | 0.808 | 2.7 | 5.1 |
| cnn_bezaug_val4_test5 | CNN | nie | 401 650 | 90 / 100 | 0.777 | 0.774 | 0.831 | 0.951 | 2.7 | 5.1 |
| gru_aug_val4_test5 | GRU | tak | 382 258 | 67 / 87 | 0.598 | 0.576 | 1.409 | 0.914 | 1.1 | 1.9 |
