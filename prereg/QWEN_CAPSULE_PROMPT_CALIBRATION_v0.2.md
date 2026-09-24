# Qwen + Capsule — kalibracja formatu promptu v0.2

**Status:** plan zamrożony przed uruchomieniem.

## Dlaczego nowa wersja

W `QWEN_VS_CAPSULE_v0.1` model zwracał `INCONCLUSIVE_NO_MATCH` dla wszystkich
pytań, również wtedy, gdy otrzymał rekord dowodowy. To jest awaria interfejsu
promptu, nie wynik o teorii TIMDR ani o kapsule. Wersja v0.1 pozostaje pełnym,
negatywnym wynikiem testu formatu.

## Co wolno zmienić

Wyłącznie komunikację instrukcji dla przypadku, w którym rekord został już
wybrany. Model, wersja wag, brak LoRA, CPU, dekodowanie deterministyczne i
brak sieci pozostają takie same. Nie używa się żadnego pytania z holdoutu v0.1.

## Kontrole syntetyczne

Trzy rekordy w `data/qwen_capsule_v0.2_calibration.json` sprawdzają:

1. odczyt identyfikatora podanego wprost;
2. odczyt prostego faktu i cytatu;
3. odmowę przy braku dowodu.

Wszystkie muszą przejść literalnie. Niepowodzenie zatrzymuje projekt przed
nowym holdoutem. Udany wynik kalibracji dowodzi wyłącznie, że mały model
potrafi odczytać ten format; nie dowodzi rozumienia TIMDR.
