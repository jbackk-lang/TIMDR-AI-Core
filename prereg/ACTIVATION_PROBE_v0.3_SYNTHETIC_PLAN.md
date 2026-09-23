# Activation probe v0.3 — plan syntetyczny przed pierwszym uruchomieniem

Status: **eksperyment eksploracyjny, nie formalny wynik TIMDR**. Ten plik i kod
definiują test przed pierwszym przebiegiem na nowych ziarnach. Nie ruszać
definicji/progów po zobaczeniu wyniku; ewentualna poprawka to osobne v0.4.
Brak commita oznacza, że sam plik nie jest niezależnym dowodem czasu zamrożenia.

## Obiekt i granice

Jednowarstwowy MLP z `neural_network.py`. Kolejne kroki = kolejne wejścia do
tego samego, zamrożonego modelu, **nie** tokeny modelu językowego. Nie
porównujemy aktywacji różnych warstw; nie używamy prawdziwej klasy w metryce.

## A — kierunek/rozkład przy porównywalnym Δ

- Trening: seed 51, 3 klasy, po 120 punktów; MLP 2→12→3, 250 epok, lr=0.1.
- Kalibracja: seed 52, 240 czystych poprawnych trajektorii długości 8,
  szum naprzemiennie 0.35/0.70. Granice koszyków Δ to kwantyle
  25/50/75/90% z ich **ostatnich** przejść aktywacji.
- Test: seed 53, 1000 trajektorii, pierwsze 200 bez przesunięcia,
  pozostałe 800 z ostatnim krokiem przesuniętym ku innej klasie o
  współczynnik jednostajny 0.3–1.25. Wynik błędny to rzeczywista
  niezgodność predykcji modelu z etykietą generatora, nie „zakłócone”.
- W każdej grupie Δ porównujemy tylko zakłócone-błędne z
  zakłóconymi-poprawnymi; grupa kwalifikuje się przy min. 10 na stronę.
- Cechy obliczone z ostatniego przejścia tej samej warstwy:
  `direction = mean(Δa)/RMS(Δa)`, `locality = max|Δa|/sum|Δa|`,
  `shape_js = JS(p(a_prev),p(a_now))`, gdzie `p=|a|/sum|a|`.
- Dla direction raportujemy podpisane mediany oraz dwustronną statystykę
  KS; dla locality i shape_js AUC w kierunku *większe u błędnych*.
  Wynik łączny to średnia statystyk wewnątrz kwalifikujących się
  koszyków, ważona liczbą `min(n_błędne,n_poprawne)` w koszyku. Nie
  porównujemy par z różnych koszyków Δ.
  Bez dostrajania kierunku po wyniku. Warunek do dalszej replikacji:
  min. 30 par w kwalifikujących się koszykach i AUC ≥ 0.60 dla co
  najmniej jednej z dwóch cech, albo KS ≥ 0.20 dla direction. To
  kryterium eksploracyjne, nie dowód przyczynowy.

## B — stabilność Λ przed pierwszym błędem

- Sekwencje długości 10, narastające przesunięcie ku innej klasie od
  kroku 2. Siła końcowa 0.1–0.45 (połowa) albo 0.7–1.35 (połowa),
  szum naprzemiennie 0.35/0.70.
- Kalibracja: seed 62, 240 sekwencji bez przesunięcia; próg dla każdej
  metryki osobno = empiryczny 95. percentyl MAKSIMUM po całej sekwencji.
  To kontroluje wielokrotne spojrzenia w czasie. Metryka kandydująca:
  `|Λ_t-Λ_(t-1)|`, gdzie Λ to znormalizowana entropia bezwzględnych
  aktywacji kanałów. Baseline: `1-max(softmax_t)` z tym samym sposobem
  kalibracji. Żadna metryka nie używa przyszłych kroków przy alarmie.
- Test: seed 63, 600 sekwencji; rzeczywista pierwsza błędna predykcja
  wyznacza `t_error`. Pierwszy alarm `t_alarm < t_error` liczy się jako
  wczesny. Błędy już w t=0 są jawnie niekwalifikowalne do wyprzedzenia.
  Sekwencje bez błędu tworzą kontrolę fałszywych alarmów.
- Kandydat na wczesny alarm wymaga: ≥30 kwalifikujących się błędów i
  ≥30 bezbłędnych sekwencji, FPR ≤ 0.15, lead-recall ≥ 0.50 i przewaga
  lead-recall nad confidence ≥ 0.10 na tych samych przypadkach.
  Raportujemy też medianę wyprzedzenia w krokach; bez zmiany progu.

## C — referencja według typu wejścia

Typ wejścia / pierwsza przewidziana klasa są tylko warstwą raportowania,
nie składnikiem metryk A/B. Nie wolno warunkować na prawdziwej etykiecie
w czasie oceny. Wyniki na syntetyce nie są wynikiem o halucynacjach ani
o rzeczywistych sieciach neuronowych.
