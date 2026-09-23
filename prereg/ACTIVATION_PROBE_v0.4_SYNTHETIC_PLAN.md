# Activation probe v0.4 — plan przed pierwszym przebiegiem

Status: nowy, eksploracyjny test syntetyczny; nie formalny wynik TIMDR,
nie test halucynacji. Kod v0.1–v0.3 i ich wyniki pozostają bez zmian.
Ten plan należy utrwalić i sprawdzić przed uruchomieniem testu. Sam
lokalny plik/hash nie jest niezależnym dowodem czasu prerejestracji.

## Korekta po v0.3

W v0.3 część B **już** używała narastającego zakłócenia; v0.4 tylko
powtarza ją na świeżym ziarnie z tymi samymi progami, nie udaje nowego
mechanizmu. `shape_js` jest symetryczną miarą zmiany rozkładu — nie
mierzy kierunku. Kierunek `mean(Δa)/RMS(Δa)` raportujemy oddzielnie.

## Wspólny model i izolacja

Ten sam zamrożony MLP 2→12→3 co v0.3: trening seed=51, 3 klasy ×120,
250 epok, lr=0.1. Analizujemy wyłącznie jego jedną warstwę ukrytą.
Etykieta generatora służy tylko do oceny błędu, nigdy do wyniku metryki.
Typ wejścia = pierwsza przewidziana klasa, wyłącznie stratyfikacja raportu.

## A — pięć poziomów Δ, świeże przypadki

- Ziarno testu 73; poziomy docelowe RMS ostatniego Δa: **1.5, 2.0,
  2.5, 3.0, 4.0**, po 300 prób na poziom. Te poziomy zostały wybrane
  przed nowym testem, powyżej granicy najwyższego przedziału v0.3
  (~1.282). Nie są kalibrowane na etykietach v0.4.
- Etykieta bazowa cyklicznie 0/1/2; szum wejścia 0.35/0.70 na przemian.
  Siedem bazowych kroków wokół centrum klasy; ostatni krok powstaje
  z przedostatniego plus losowy kierunek jednostkowy 2D. Długość
  przesunięcia wyznaczamy 32 krokami bisekcji w zakresie [0,20],
  używając **tylko** RMS zmiany aktywacji. Bez zaglądania w etykietę
  wyniku i bez strojenia wielkości Δ po teście.
- Gdy poziom jest nieosiągalny w [0,20] lub błąd dopasowania Δ jest
  większy niż 0.02, przypadek oznaczamy jako technicznie wykluczony;
  liczby wykluczeń raportujemy. Nie dobieramy zamiennika.
- Dla każdego poziomu osobno porównujemy błędne i poprawne końcowe
  predykcje. Kwalifikacja poziomu: ≥20 na stronę. Raportujemy AUC
  `shape_js`, AUC samego Δ i AUC confidence (`1-max(softmax)`), zawsze
  w z góry określonym kierunku „wyższe u błędnych”. Raportujemy też
  podpisane mediany direction i dwustronny KS direction. Bez wyboru
  znaku kierunku po wyniku.
- Dodatkowo stratyfikujemy AUC `shape_js` według pierwszej przewidzianej
  klasy; typ kwalifikuje się przy ≥20 błędnych i ≥20 poprawnych.
- Kandydat do **dalszej, realnej** replikacji wymaga ≥3 kwalifikujących
  się poziomów i ≥2 typów, AUC `shape_js` ≥0.65 w każdym z tych
  poziomów oraz ≥0.60 w co najmniej 2 typach. W każdym kwalifikującym
  się poziomie AUC `shape_js` musi przewyższać AUC samego Δ o ≥0.05.
  To nie upoważnia do nazwania metryki „operatorem kierunkowym”.

## B — niezależne powtórzenie narastającego zakłócenia

- Nowy test seed=64, 600 sekwencji po 10 kroków; generator i podział
  siły 0.1–0.45 / 0.7–1.35 identyczny z v0.3.
- Wykorzystujemy bez ponownej kalibracji progi v0.3:
  Λ `0.1488384562221906`, confidence `0.10808176126798885`.
  Alarm tylko z bieżących i wcześniejszych kroków, nigdy z przyszłości.
- Raport: liczba błędów z t_error>0, liczba bez błędu, lead-recall,
  FPR oraz mediana wyprzedzenia. Warunek kandydata jak w v0.3:
  ≥30 kwalifikujących się błędów, ≥30 bezbłędnych, FPR≤0.15,
  lead-recall Λ≥0.50 i przewaga nad confidence ≥0.10.

Nie wolno zmienić poziomów, progów, liczebności ani kryteriów po
obejrzeniu wyniku. Niepomyślny wynik jest pełnoprawnym wynikiem.
