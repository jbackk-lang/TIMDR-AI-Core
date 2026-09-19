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

Osobna, nowa hipoteza używa 16 cech z przebiegów EMT (dla ośmiu lokalizacji:
szczytowy prąd RMS i minimalne napięcie RMS, oba znormalizowane do pierwszych
200 ms). Nie używa `sc_type`, lokalizacji uszkodzenia ani czasu zdarzenia jako
cech. Ponieważ to inna hipoteza, ma własną prerejestrację:

```powershell
.\run.bat --protect90-waveform-freeze "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
.venv\Scripts\python.exe -m pip install -e ".[waveform]"
.\run.bat --protect90-waveform "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
```

## Granica odpowiedzialności

Status `SUPPORTED`, `NOT_SUPPORTED` lub `INCONCLUSIVE` wynika z
prerejestrowanego testu i kontroli dostarczonych przez uruchomienie domenowe.
Warstwy AI/LTR mogą przetwarzać reprezentacje, lecz nie mogą samodzielnie
ustanowić dowodu ani wyniku empirycznego.

## Lekki lokalnie, otwarty na źródła zewnętrzne

Domyślnie środowisko wykonuje jeden proces naraz, nie zapisuje kopii przebiegów
i odrzuca epizody większe niż 8 MB. Materiał z internetu może zostać pobrany
wyłącznie na jawne polecenie, przez HTTPS, z wcześniej podaną sumą SHA-256 i
limitem 25 MB. Nie ma automatycznego crawlowania, wysyłania danych, wykonywania
obcego kodu ani samoczynnego uczenia się na pobranych plikach.

Szablon [external_sources.template.json](external_sources.template.json)
jest miejscem na deklarację źródła. Po uzupełnieniu można pobrać jedną pozycję:

```powershell
.\run.bat --external-source "external_sources.json" "source-id"
```

Plik trafi do lokalnego, ignorowanego przez Git `external_cache/`. Dopiero nowa
prerejestracja może nadać mu status danych do analizy lub uczenia.
