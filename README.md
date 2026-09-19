# TIMDR-AI-Core

Mały, niezależny rdzeń protokołu TIMDR jako **filtr epistemiczny**.

Nie jest to model matematyczny TIMDR ani silnik, który sam ustala wyniki
empiryczne. Zapewnia natomiast powtarzalne zasady:

- kanoniczną prerejestrację z odciskiem SHA-256;
- jawne kontrole dodatnią i ujemną;
- blokadę werdyktu `SUPPORTED` bez kontroli i dostarczonej evidencji testu;
- rozdzielenie potoku Λ–τ–ρ od oceny statusu hipotezy.

## Uruchomienie

```powershell
python timdr_ai_core.py
python -m pytest -q
```

`pytest` jest potrzebny tylko do uruchomienia testów. Sam moduł korzysta
wyłącznie z biblioteki standardowej Pythona.

## PROTECT-90: uczenie badawcze

Pierwszy plan uczenia jest czteroklasową klasyfikacją `sc_type` w zbiorze
PROTECT-90. Źródłem jest symulacja EMT, a nie pomiar polowej sieci energetycznej.
To transparentny baseline najbliższego centroidu na zamrożonych metadanych
scenariusza — nie algorytm zabezpieczenia i nie wynik B4.

Najpierw, **przed pierwszym fit**, zamraża się prerejestrację i jej podział
stratyfikowany: 60% trening, 20% kalibracja, 20% nietykalny holdout.

```powershell
.\run.bat --protect90-freeze "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
.\run.bat --protect90 "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
```

Druga komenda używa jedynie treningu i kalibracji. Raport z uruchomienia jest
celowo zapisywany do ignorowanego przez Git `learning_runs/`; nie należy
otwierać holdoutu w celu doboru cech, modelu ani parametrów.

## Granica odpowiedzialności

Status `SUPPORTED`, `NOT_SUPPORTED` lub `INCONCLUSIVE` wynika z
prerejestrowanego testu i kontroli dostarczonych przez uruchomienie domenowe.
Warstwy AI/LTR mogą przetwarzać reprezentacje, lecz nie mogą samodzielnie
ustanowić dowodu ani wyniku empirycznego.
