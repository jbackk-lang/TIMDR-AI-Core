# Claim Router — trening v0.1

**Status:** zamrożone train/calibration/holdout przed pierwszym treningiem.

## Zadanie

Lekki klasyfikator uczy się wyłącznie mapy `pytanie -> identyfikator węzła`
z `claim_router_dataset_v0.1.json`. Nie uczy się statusu, faktu ani tekstu
odpowiedzi. Te pozostają własnością deterministycznego Claim Graph.

## Bramka logiczna

Predykcja klasyfikatora jest obserwacją pomocniczą. Werdykt pochodzi tylko z
`claim_graph_gate.decide()`. Gdy predykcja nie zgadza się z grafem, raport
oznacza rozjazd, lecz odpowiedź pozostaje decyzją grafu. Pytanie poza grafem
zawsze otrzymuje `INCONCLUSIVE_NO_MATCH`.

## Rozdział danych

- `train`: uczenie licznika słów;
- `calibration`: jedyny dostępny etap oceny przed holdoutem;
- `holdout`: nie może zostać odczytany przez komendę treningową.

Każda zmiana zestawu, tokenizacji, algorytmu lub progu po calibration wymaga
nowej wersji planu. Wysoka trafność routingu nie jest dowodem TIMDR; mierzy
jedynie rozpoznawanie pytań w tej małej, opisanej ontologii.
