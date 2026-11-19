# Test ciszy na foldzie 4

Nagrania z co najmniej 20% ciszy: 90 z 400 (średnio 55% ciszy). Cisza zastąpiona szumem ok. -50 dBFS.

| Model | Accuracy (cały fold) | Accuracy: nagrania z ciszą | Po zamianie cisza → szum | Zmienione predykcje |
|---|---|---|---|---|
| cnn_bezaug_lr0.001_cosine_val4_test5 | 0.772 | 0.711 | 0.511 | 48.9% |
| cnn_aug_lr0.001_cosine_val4_test5 | 0.760 | 0.689 | 0.711 | 17.8% |
| gru_aug_lr0.001_cosine_val4_test5 | 0.608 | 0.611 | 0.622 | 3.3% |
