# Porównanie wariantów na 4 podziałach (test = fold 5 nieużywany)

## Wyniki na każdym podziale (val acc)

| Wariant | val = fold 1 | val = fold 2 | val = fold 3 | val = fold 4 |
|---|---|---|---|---|
| cnn_aug_lr0.001_cosine | 0.715 | 0.715 | 0.733 | 0.760 |
| cnn_bezaug_lr0.001_cosine | 0.755 | 0.730 | 0.735 | 0.772 |

## Średnia ± odchylenie standardowe

| Wariant | Val acc | Val F1 macro | Val strata | Train acc (najl. epoka) | Różnica trening–walidacja |
|---|---|---|---|---|---|
| cnn_aug_lr0.001_cosine | 0.731 ± 0.021 | 0.722 ± 0.024 | 0.994 ± 0.105 | 0.829 ± 0.015 | 0.099 ± 0.034 |
| cnn_bezaug_lr0.001_cosine | 0.748 ± 0.020 | 0.741 ± 0.022 | 0.933 ± 0.066 | 0.951 ± 0.005 | 0.203 ± 0.018 |

**cnn_bezaug_lr0.001_cosine − cnn_aug_lr0.001_cosine** (val acc, ten sam podział): +4.0 p.p., +1.5 p.p., +0.2 p.p., +1.2 p.p. | średnio +1.7 p.p. | lepszy w 4 z 4 podziałów
