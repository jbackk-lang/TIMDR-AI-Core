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

## Operatory T/M/R: realna matematyka wpięta w warstwy LTR

`timdr_operators.py` dostarcza gotowe, wiernie przeniesione (1:1, z podanym
źródłem) funkcje do wpięcia w `LayerTTopology`/`LayerMModal` przez ich
argument `transform=` — same warstwy w `timdr_ai_core.py` zostają domyślnie
identycznością, tak jak wcześniej. `FundamentalModelLTR` przekazuje T i M
równolegle to samo surowe wejście (patrz niżej), więc oba operatory można
wpiąć naraz na tym samym oknie:

- `fft_dominant_mode(window, fs)` — ekstrakcja dominującego trybu (f, phi, A)
  z okna, port z `TIMDR-Modal-Formalism/timdr_modal/real_data_validation.py`;
- `winding_number`/`crossing_number` — niezmienniki topologiczne krzywej
  osadzonej z opóźnieniem, port z
  `GIA-TIMDR/core/winding_crossing_ms_bridge.py` (uczciwy zakres przeniesiony
  z repo źródłowego: odrzucone na danych syntetycznych, silny-lecz-częściowy
  sygnał na realnych łożyskach CWRU, niespójny na sejsmice/BTC — diagnostyczne,
  nie selekcyjne);
- `Modality`/`is_resonant` — Aksjomat 5 gałęzi K, port z
  `TIMDR-Modal-Formalism/timdr_modal/phase_sync.py` (bez numpy).

```powershell
.venv\Scripts\python.exe -m pip install -e ".[operators]"
python -m pytest -q tests/test_timdr_operators.py
```

Przykład wpięcia jednej warstwy:

```python
from timdr_ai_core import LayerMModal
from timdr_operators import fft_dominant_mode

layer = LayerMModal(transform=lambda w: fft_dominant_mode(w, fs=500.0))
```

**Przepływ danych w `FundamentalModelLTR`:** T i M czytają surowe wejście
równolegle i bezpośrednio (oba widzą to samo okno). `I` scala ich dwie
reprezentacje — `transform` dostaje jeden słownik `{"T":..,"M":..}`. `It`
i `R` przetwarzają dalej sekwencyjnie. `E` na końcu łączy T/M/R — `transform`
dostaje `{"T":..,"M":..,"R":..}`. `forward()` zwraca pełny słownik stanu
wszystkich warstw: `{"T","M","I","It","R","E"}`, nie tylko wyjście `E`.
Model z samymi warstwami domyślnymi (identyczność) zwraca
`{"T": raw, "M": raw, "R": raw}` opakowane identycznością `E`, nie surowe
wejście bez zmian. Test:
`tests/test_timdr_operators.py::test_fundamental_model_ltr_can_now_combine_t_and_m_on_the_same_raw_window`
pokazuje `winding_number`/`crossing_number` (T) i `fft_dominant_mode` (M)
działające razem na tym samym realnym oknie w jednym wywołaniu.

**Ograniczenie `crossing_number`:** wersja zwektoryzowana trzyma w pamięci
macierz (n, n) wszystkich par segmentów — O(n²) pamięci. Funkcja odmawia
okien dłuższych niż `max_length` (domyślnie 3000, ~1,3 GB w najgorszym razie)
z czytelnym `TimdrOperatorsError` zamiast próbować alokację i zawieść
surowym `MemoryError` (realne okno Paderborn, 64000 próbek @ 64 kHz,
wymagałoby ~65 GB). Dłuższe realne okna trzeba samodzielnie zdownsamplować
lub podzielić na fragmenty przed wywołaniem. Weryfikacja end-to-end na
zamrożonym, autoryzowanym oknie treningowym Paderborn (bez dotykania
holdoutu):

```powershell
.venv\Scripts\python.exe -m pip install unrar-cffi scipy numpy
python examples\run_timdr_operators_on_paderborn.py
```

## Lekki lokalnie, uczący się z sieci

Lokalny komputer ma wykonywać mało pracy: jeden proces naraz, bez kopii
przebiegów i z limitem 8 MB na epizod. Rozwój źródłowy może zachodzić
automatycznie w sieci: środowisko cyklicznie pobiera nowe dokumenty z
zadeklarowanych katalogów HTTPS i aktualizuje lokalny graf pochodzenia oraz
słownik pojęć. Jeden cykl jest ograniczony do 16 dokumentów po 2 MB.

Automatyczne uczenie sieciowe nie pobiera ani nie uruchamia obcego kodu, nie
wysyła danych lokalnych, nie modyfikuje prerejestracji i nie ustanawia wyniku
TIMDR. Aktualizuje wyłącznie model-kandydata wiedzy o źródłach.

Skopiuj `online_catalogs.template.json`, wpisz katalogi zwracające JSON w
formacie `items: [{id, url, title}]`, a następnie uruchom cykl:

```powershell
.\run.bat --online-learn "online_catalogs.json"
```

Stan uczenia zapisuje się w ignorowanym przez Git `external_cache/`.

Można też uszeregować zebrane dokumenty jako kandydatów czterech niezależnych
gałęzi TIMDR — M/S, G, K i META-DYNAMICS. Ranking jest wyłącznie trasowaniem
źródeł po słowach kluczowych; nie tworzy mostu między gałęziami i nie stanowi
wyniku badawczego:

```powershell
.\run.bat --online-rank
```

Po rankingu można zbudować kolejkę badawczą. Każde źródło trafia dokładnie do
jednej najlepiej pasującej gałęzi i otrzymuje listę brakujących artefaktów;
żadne źródło nie staje się przez to automatycznie datasetem, hipotezą ani
mostem między gałęziami.

```powershell
.\run.bat --research-queue
```

### Pobranie zamrożonego artefaktu

Niezależnie od automatycznego uczenia można pobrać konkretny artefakt danych,
gdy jego SHA-256 jest znane z góry. Służy do tego
`external_sources.template.json`.

Szablon [external_sources.template.json](external_sources.template.json)
jest miejscem na deklarację źródła. Po uzupełnieniu można pobrać jedną pozycję:

```powershell
.\run.bat --external-source "external_sources.json" "source-id"
```

Plik trafi do lokalnego, ignorowanego przez Git `external_cache/`. Dopiero nowa
prerejestracja może nadać mu status danych do analizy lub uczenia.

## Paderborn: zewnętrzna replikacja M/S

Pobrany minimalny kandydat Paderborn zawiera trzy klasy: zdrowe `K001`,
sztuczne EDM uszkodzenie pierścienia zewnętrznego `KA01` oraz pierścienia
wewnętrznego `KI01`. Archiwa surowych danych są ignorowane przez Git.

Zamrożenie selekcji nie czyta wartości sygnału — używa wyłącznie nazw plików
MATLAB w archiwach i tworzy podział 144 / 48 / 48:

```powershell
.\run.bat --paderborn-freeze
```

Przed ekstrakcją przebiegów potrzebna jest osobna prerejestracja hipotezy M/S.

Taka prerejestracja została zapisana jako `PADERBORN_MS_HYPOTHESIS_v0.1`.
Definiuje kanał `vibration_1`, cztery jednosekundowe okna na pomiar, cztery
cechy M/S, kontrolę dodatnią i ujemną oraz jednorazową ocenę holdoutu. Przed
implementacją ekstraktora nie należy otwierać payloadów MATLAB z holdoutu.

Ekstraktor Paderborn otwiera wyłącznie zamrożone pliki treningowe i
kalibracyjne, a żądanie holdoutu odrzuca przed otwarciem archiwum. Techniczny
odczyt jednego pliku treningowego:

```powershell
.\run.bat --paderborn-schema
```
