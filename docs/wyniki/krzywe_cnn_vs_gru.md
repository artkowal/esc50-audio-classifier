# Zestawienie treningów (walidacja)

| Eksperyment | Model | Augmentacja | Parametry | Najlepsza epoka | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Czas epoki [s] | Łącznie [min] |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn_bezaug_lr0.001_cosine_val4_test5 | CNN | nie | 401 650 | 90 / 100 | 0.772 | 0.768 | 0.839 | 0.951 | 2.8 | 5.2 |
| gru_aug_lr0.001_cosine_val4_test5 | GRU | tak | 382 258 | 67 / 87 | 0.608 | 0.594 | 1.393 | 0.910 | 1.1 | 1.9 |
