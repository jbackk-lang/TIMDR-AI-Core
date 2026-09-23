# Activation probe v0.9 — wynik tylko na calibration (UCI HAR)

Status: **CALIBRATION ONLY; brak potwierdzenia `shape_js` na realnych danych**.
Oficjalnych członków `test/` nie otwarto. Nie zmieniono modelu, podziału,
cech ani progów po obejrzeniu wyniku. Źródło: [UCI HAR](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones),
DOI 10.24432/C54S4K, CC BY 4.0. Hash pobranego ZIP (SHA-256):
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.

Plan v0.8 zatrzymał się **przed odczytem próbek** z powodu blokady
`scikit-learn` przez Windows Device Guard. Zgodnie z planem v0.9
wykonano jeden przebieg siecią NumPy 561→64→6 z repo, 500 epok,
bez early stopping. Python 3.12.10, NumPy 2.5.3.

| Miara | Wynik calibration |
| --- | ---: |
| Osoby fit / calibration | 16 / 5 |
| Okna fit / calibration | 5655 / 1697 |
| Sąsiednie pary tej samej osoby i aktywności | 1635 |
| Błędne / poprawne pary | 193 / 1442 |
| Osoby z błędem / poprawną parą | 5 / 5 |
| Trafność okien fit / calibration | 0,9873 / 0,8821 |
| AUC `shape_js` | **0,4861** |
| AUC `delta_rms` | 0,5588 |
| AUC `uncertainty` (baseline) | **0,7093** |

Kontrola techniczna i bramka liczebności przeszły. W kwartylach Δ
(`shape_js` AUC, od najmniejszego do największego Δ): **0,6060;
0,5260; 0,4537; 0,3315**. Każdy kwartyl miał ≥10 błędnych i ≥10
poprawnych par. Najwyższy kwartyl odwrócił przewidywany kierunek.

Interpretacja: na tej kalibracji `shape_js` **nie wykazuje użytecznej
separacji błędów** i wypada gorzej niż zwykła niepewność klasyfikatora.
To nie jest wynik potwierdzający ani dowód, że operator nigdy nie
zadziała. Pary okien są skorelowane, a brak jawnych znaczników czasu
ogranicza interpretację chronologiczną. Nie ma podstaw do twierdzenia,
że dodaje informację ponad confidence. Nie próbujemy odzyskać dodatniego
wyniku przez tuning na tych samych osobach. Holdout pozostaje zamknięty.

Odtwarzanie tylko train/calibration:

```powershell
$env:OPENBLAS_NUM_THREADS='2'
$env:OMP_NUM_THREADS='2'
.\.venv\Scripts\python.exe activation_probe_uci_har.py C:\Users\jback\Downloads\a\data\uci_har_240.zip
```
