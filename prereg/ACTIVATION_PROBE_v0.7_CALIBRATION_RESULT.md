# Activation probe v0.7 — wynik bramki kalibracyjnej

Status: **INCONCLUSIVE — brak mocy do diagnozowania błędów**.
Odczytano wyłącznie train i calibration według planu v0.7;
`holdout_accessed=False`. Plan v0.7 nie został jeszcze zatwierdzony
commitem, więc blokada kodowa i tak odmawia otwarcia holdoutu.

- Train: 144 całe pomiary, 1152 półsekundowych okien. Trafność modelu
  na train: 1,000 (informacja pomocnicza, nie wynik testu).
- Calibration: 48 całych pomiarów, 336 przejść między oknami.
  Błędnych predykcji 0; poprawnych 336. Wymagano ≥30 każdej grupy
  i ≥5 różnych pomiarów każdej grupy. Bramka nie przeszła.
- Korekta długości na train+calibration: trzy pomiary miały niedobór
  1–2 próbek; 171 miało nadmiar odcinany po 4 s. Podpisana korekta
  `len-256000` mieściła się od -2 do +21049. Pełna lista per pomiar
  jest odtwarzalna z `calibration_report()`.

Nie trenowano logistycznych diagnostyk błędu, nie wyznaczano AUC
`shape_js`, nie kalibrowano alarmu na poprawnych przykładach i nie
otwierano holdoutu. **Nie wolno** dobierać słabszego modelu albo
etykiet po obejrzeniu tego wyniku i nazywać tego kontynuacją v0.7.
Nowy problem badawczy wymaga osobnego planu i nowego zamrożenia.
