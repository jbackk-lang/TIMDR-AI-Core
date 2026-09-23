# v0.5 — zatrzymane przed kalibracją i holdoutem

Pierwszy pomiar części **train** (`K001/N15_M01_F10_K001_6.mat`) ma
256575 próbek kanału `vibration_1`, podczas gdy plan v0.5 dopuszczał
wyłącznie 256000 lub 256001. Istniejący ekstraktor odmówił odczytu.
Nie policzono cech dla całego train, nie uczono modelu, nie czytano
calibration ani holdoutu. v0.5 nie daje wyniku empirycznego.

Nie zmieniamy tego planu po fakcie. Nowa reguła techniczna musi być
zaplanowana osobno jako v0.6 przed następną próbą.
