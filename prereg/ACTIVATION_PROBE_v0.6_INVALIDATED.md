# v0.6 — zatrzymane na train, bez kalibracji modelu

Reguła v0.6 obejmowała 256000–262400 próbek. W części train
napotkano `KA01/N15_M07_F04_KA01_5.mat` z 255999 próbkami, więc
ekstraktor przerwał przed uczeniem modelu i przed oceną calibration.
Holdout nie został otwarty.

Po zatrzymaniu wykonano wyłącznie audyt **długości** wszystkich
144 członków train i 48 calibration, bez uczenia lub testowania
metryki: train min=255998, max=273851; calibration min=256000,
max=277049. To dane projektowe do nowej wersji v0.7, nie wynik
diagnostyczny. Lista per pomiar może zostać odtworzona funkcją
`audit_train_calibration_lengths()`.
