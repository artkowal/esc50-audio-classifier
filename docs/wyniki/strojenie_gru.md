# Porównanie wariantów na 4 podziałach (test = fold 5 nieużywany)

## Wyniki na każdym podziale (val acc)

| Wariant | val = fold 1 | val = fold 2 | val = fold 3 | val = fold 4 |
|---|---|---|---|---|
| gru_aug_lr0.001_cosine | 0.598 | 0.550 | 0.615 | 0.608 |
| gru_bezaug_lr0.001_cosine | 0.560 | 0.512 | 0.575 | 0.535 |
| gru_aug_lr0.001_cosine_do0.5_gd0.3 | 0.568 | 0.588 | 0.590 | 0.603 |

## Średnia ± odchylenie standardowe

| Wariant | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Różnica trening–walidacja |
|---|---|---|---|---|---|
| gru_aug_lr0.001_cosine | 0.593 ± 0.029 | 0.584 ± 0.025 | 1.493 ± 0.125 | 0.886 ± 0.037 | 0.293 ± 0.057 |
| gru_bezaug_lr0.001_cosine | 0.546 ± 0.028 | 0.534 ± 0.029 | 1.682 ± 0.096 | 0.868 ± 0.050 | 0.323 ± 0.023 |
| gru_aug_lr0.001_cosine_do0.5_gd0.3 | 0.587 ± 0.014 | 0.576 ± 0.011 | 1.565 ± 0.119 | 0.870 ± 0.004 | 0.283 ± 0.014 |
