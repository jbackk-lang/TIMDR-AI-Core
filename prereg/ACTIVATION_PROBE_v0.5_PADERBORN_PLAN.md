# Activation probe v0.5 — Paderborn, plan przed odczytem nowego holdoutu

Status: diagnostyka jednej warstwy sieci na **rzeczywistym sygnale drgań**,
ale nie formalny wynik TIMDR, nie detektor halucynacji i nie walidacja
na obrazach. Wynik v0.4 był syntetyczny; żadnego progu nie przenosimy.
Ten plan powinien mieć osobny lokalny commit **przed** uruchomieniem
oceny holdoutu. Istniejące archiwa i poprzedni podział są znane, więc
to replikacja nowej metryki na nowej domenie, nie idealnie dziewiczy
zbiór danych dla całego ekosystemu repo.

## Dane i jednostka niezależności

- Paderborn Bearing Data Center, lokalne archiwa `K001.rar` (healthy),
  `KI01.rar` (inner ring EDM), `KA01.rar` (outer ring EDM), źródło:
  https://groups.uni-paderborn.de/kat/BearingDataCenter/ .
- Kanał `vibration_1`, fs=64000 Hz. Używamy **dokładnie** członków
  archiwów z `prereg/PADERBORN_MS_REPLICATION_v0.1.json`: 144 pomiary
  train, 48 calibration, 48 holdout, po równo na klasę. Podział na
  całych plikach pomiarowych, a nie oknach. Nie przenosimy okien tego
  samego pomiaru między zbiorami.
- Każde 4 s dzielimy chronologicznie na 8 niepokrywających się okien
  po 0,5 s (32000 próbek). `+1` próbka, jeśli występuje, jest
  odrzucona na końcu zgodnie z istniejącym ekstraktorem. Jedna
  trajektoria = 8 kolejnych okien tego samego pomiaru. Ocena błędu
  dotyczy kroków 2–8, 7 przejść na pomiar. Kroki nie są niezależnymi
  replikacjami; przedziały ufności resamplują **całe pomiary**.

## Model i wejście

- Osiem cech z każdego okna, ustalonych bez czytania holdoutu:
  `log1p(RMS)`, `log1p(peak/RMS)`, `log1p(kurtosis_nonexcess)`,
  `log1p` energii widmowej w pasmach [0,500), [500,2000),
  [2000,8000), [8000,16000), [16000,32000] Hz. DC włączony.
  Dla zerowego RMS wartości ilorazowe wynoszą 0. Wszystkie cechy
  muszą być skończone. Normalizacja średnia/std z train tylko.
- MLP 8→12 ReLU→3 softmax z `neural_network.py`, seed=510,
  full-batch, 300 epok, lr=0.05. Klasy w kolejności K001=0,
  KI01=1, KA01=2. Model zamrożony po train; kalibracja nie dopasowuje
  jego wag. Tylko aktywacje tej samej warstwy ukrytej.
- Rozmiar błędu: czy predykcja bieżącego okna różni się od znanej
  etykiety pomiaru. Etykieta nie wchodzi do score. Diagnostyka może
  być oceniona tylko przy ≥30 błędnych i ≥30 poprawnych przejściach
  holdoutu, obejmujących ≥5 różnych pomiarów w obu grupach.

## Cechy diagnostyczne i dwa porównania

- `shape_js`: dokładnie implementacja v0.3, symetryczna dywergencja
  Jensena–Shannona między rozkładami `|a_{t-1}|/sum|a_{t-1}|` i
  `|a_t|/sum|a_t|`. To **kształt**, nie kierunek.
- `delta_rms`: RMS(a_t-a_{t-1}); `uncertainty=1-max(softmax_t)`.
- Cztery koszyki Δ według kwartylów z **kalibracji**, bez etykiet
  błędu. W każdym koszyku holdoutu z ≥10 błędnymi i ≥10 poprawnymi
  podajemy AUC `shape_js`, `delta_rms` i `uncertainty`; nie łączymy
  par z różnych koszyków. Mniej niż 3 kwalifikujące koszyki to
  brak dowodu stabilności zakresowej, nie licencja na zmianę granic.
- Przyrost ponad baseline: na calibration uczymy dwa z góry określone
  modele logistyczne błędu: bazowy `[uncertainty,delta_rms]` i
  rozszerzony `[uncertainty,delta_rms,shape_js]`. Każdy standaryzuje
  swoje cechy średnia/std z calibration; skala 0 zastąpiona 1.
  Pełny batch gradient descent, seed niepotrzebny, wagi początkowe 0,
  500 kroków lr=0.1, L2=1.0 dla wag (nie interceptu).
- Test główny na holdout: ΔAUC = AUC(rozszerzony) - AUC(bazowy).
  95% przedział percentylowy z 2000 bootstrapów po **48 całych
  pomiarach**, seed=515. Repliki bez obu klas błędu pomijamy i
  raportujemy ich liczbę. Warunek dodatkowej wartości: ΔAUC≥0.03,
  dolna granica CI>0 oraz ≥3 koszyki z AUC `shape_js`≥0.60.

## Bramka jakości i kontrole

- Kontrola techniczna (+): znane przykłady wektorów aktywacji muszą
  dać `shape_js=0` dla niezmiennego rozkładu i `shape_js>0` dla
  zmienionego rozkładu; test jednostkowy przed holdoutem.
- Bramka mocy: na calibration ≥30 błędnych i ≥30 poprawnych przejść
  z ≥5 pomiarów na grupę. Jeśli nie, NIE otwieramy holdoutu w tym
  protokole i raportujemy `INCONCLUSIVE`.
- Kontrola negatywna: dla niezakłóconych, poprawnych przejść holdoutu
  wskaźnik alarmów `shape_js` po progu 95. percentyla poprawnych
  przejść calibration powinien być ≤0.15. Próg wyznacza calibration
  przed holdoutem. Jeśli kontrola nie przechodzi, nie nazywamy wyniku
  działającą diagnostyką nawet gdy AUC/ΔAUC jest wysokie.
- Gdy skrypt nie może odczytać źródła, klasy/kształty są błędne lub
  bramka mocy nie przechodzi, stan `INCONCLUSIVE`; nie zmieniamy
  parametrów po fakcie. Wynik negatywny zachowujemy bez retuszu.

Nie publikujemy surowych danych. Wynik nie ustala statusu żadnego
mostu TIMDR, tylko przydatność lokalnej diagnostyki aktywacji.
