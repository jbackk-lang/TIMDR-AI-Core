# Activation probe v0.6 — jawna poprawka długości pomiaru

Status: zastępuje wyłącznie techniczną regułę długości z v0.5,
zatrzymanej na pierwszym pomiarze train. Pozostałe definicje, split,
model, cechy, progi, metryki, bootstrap i kryteria sukcesu są
**dosłownie takie jak w** `ACTIVATION_PROBE_v0.5_PADERBORN_PLAN.md`.
Nie otwarto calibration ani holdoutu v0.5.

Nowa reguła ekstrakcji: kanał `vibration_1` musi mieć co najmniej
256000 i najwyżej 262400 próbek (4 s + maks. 0,1 s przy 64 kHz).
Deterministycznie używa się **pierwszych 256000 próbek**; końcowy
nadmiar jest logowany jako liczba obciętych próbek dla każdego
pomiaru. Niedobór lub większy nadmiar to błąd techniczny i przerwanie
przed testem. Reguła nie zależy od etykiety, sygnału ani wyniku
modelu. Osiem okien po 32000 próbek pozostaje bez zmian.

Najpierw uruchomienie train/calibration i bramka mocy. Do testu
holdoutu wolno przejść wyłącznie gdy ten **dokładny** plik planu
został zatwierdzony jako osobny commit Git przed odczytem holdoutu.
W przeciwnym razie kod ma odmówić dostępu. Brak tuningu po wyniku.
