# Porównanie modeli CNN i GRU

## Koszt obliczeniowy

| | CNN | GRU |
|---|---|---|
| Parametry | 401 650 | 382 258 |
| Rozmiar checkpointu [MB] | 1.89 | 1.54 |
| Czas epoki treningu [s] | 2.81 | 1.15 |
| Łączny czas treningu [min] | 5.3 | 1.9 |
| Predykcja GPU, batch 32 [ms / nagranie] | 0.841 | 0.431 |
| Predykcja CPU, pojedyncze nagranie [ms] | 16.1 | 49.7 |
| Pamięć GPU: krok treningu, batch 32 [GB] | 1.42 | 0.85 |
| Pamięć GPU: predykcja, batch 32 [GB] | 0.56 | 0.29 |

## Rozdzielenie klas w przestrzeni cech (test, silhouette, cosinus)

| Reprezentacja | Silhouette (50 klas) |
|---|---|
| MFCC (EDA) | -0.105 |
| CNN (cechy sieci) | 0.024 |
| GRU (cechy sieci) | 0.040 |

Silhouette: od −1 do 1; wyżej = nagrania tej samej klasy leżą bliżej siebie niż innych klas.

## Porównanie z literaturą (modele uczone tylko na ESC-50)

| Model | Accuracy |
|---|---|
| Prosta CNN na spektrogramie (Piczak, 2015) | 64,5% |
| Sieci na surowym sygnale (zależnie od głębokości) | 51,1–68,5% |
| Człowiek | 81,3% |
| **Nasz CNN** (5-krotna walidacja krzyżowa) | **0,734 ± 0,033** |
| **Nasz GRU** (5-krotna walidacja krzyżowa) | **0,583 ± 0,027** |
