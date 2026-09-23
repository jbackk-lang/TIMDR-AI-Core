# Activation probe v0.7 — reguła czasu po audycie train/calibration

Status: zastępuje tylko regułę długości z v0.5/v0.6. Wszelkie inne
elementy projektu, modelu, cech, podziału, testów i kryteriów są
identyczne z `ACTIVATION_PROBE_v0.5_PADERBORN_PLAN.md`. v0.5 i v0.6
są jawnie oznaczone jako zatrzymane przed kalibracją modelu; żadnego
wyniku diagnostycznego z nich nie uzyskano.

Audyt metadanych train (144 pomiary) i calibration (48 pomiarów), bez
otwierania holdoutu, wykazał długości train 255998–273851 oraz
calibration 256000–277049. Na tej podstawie przed dalszym przebiegiem
ustala się **jedną** regułę niezależną od etykiety i wartości sygnału:

1. Kanał `vibration_1` musi mieć od 255998 do 320000 próbek. Poza
   tym zakresem wynik `INCONCLUSIVE` technicznie, bez poprawki po
   odczycie holdoutu.
2. Zawsze używa się pierwszych 256000 próbek jako 4 sekund przy
   fs=64000. Nadmiar za tym horyzontem się odcina. Brakujące 1 lub 2
   próbki na końcu uzupełnia się zerami. Żadnego interpolowania lub
   rozciągania całego przebiegu.
3. Liczba oryginalnych próbek oraz podpisana korekta `len-256000`
   są logowane dla **każdego** pomiaru; wartość ujemna oznacza padding.
   Cztery sekundy dzieli się następnie na 8 okien po 32000 próbek.

Zasada 320000 jest limitem technicznym jednej dodatkowej sekundy;
nie została wyprowadzona z wyników modelu. Jeśli holdout go przekroczy,
test się zatrzyma i nie wolno przesuwać limitu pod ten sam holdout.

Tak jak w v0.5, test holdoutu wymaga osobnego commita tego dokładnego
planu **przed** odczytem. Wynik train/calibration może służyć do
ustalenia, czy test ma moc; zmiana metody po kalibracji wymaga następnej
wersji i ponownego zamrożenia. Wynik na calibration nie jest wynikiem
potwierdzającym.
