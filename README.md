# TIMDR-AI-Core

Python, biblioteka standardowa + opcjonalnie numpy/pandas/scipy/rarfile.
Wymaga Python 3.10+.

## Zakres w jednym akapicie

`TIMDRProtocol` (`timdr_ai_core.py`) jest filtrem epistemicznym: sam nie
ustala żadnego wyniku empirycznego, tylko ocenia dostarczoną z zewnątrz,
prerejestrowaną ewidencję i kontrole (`run_test()`), blokując werdykt
`SUPPORTED` bez obu. Reszta repozytorium to: (1) generyczny, wymienny potok
reprezentacji `FundamentalModelLTR` (warstwy T/I/M/It/R/E), (2) realne,
przeniesione z sióstr-repo operatory matematyczne do wpięcia w te warstwy,
(3) moduły uczenia, które faktycznie trenują klasyfikatory i liczą realne
metryki na danych (najbliższy centroid, mały MLP) — to prawdziwe uczenie
maszynowe, nie atrapa — ale ich wynik jest zawsze kandydatem badawczym,
nigdy bezpośrednim wejściem do werdyktu protokołu. Zobacz "Granica
odpowiedzialności" niżej dla dokładnego podziału.

## Uruchomienie

```powershell
.\run.bat --tests
```

Tworzy `.venv`, instaluje `pytest` i uruchamia cały zestaw testów. Bez
argumentu `run.bat` odpala krótkie demo protokołu
(`examples\protocol_demo.py`). Ręcznie, bez batcha:

```powershell
python timdr_ai_core.py
python -m pytest -q
```

Rdzeń (`timdr_ai_core.py`) korzysta wyłącznie z biblioteki standardowej.
Numpy/pandas/scipy/rarfile są potrzebne tylko dla konkretnych, opcjonalnych
ścieżek opisanych niżej (`pip install -e ".[nazwa-extra]"`).

## TIMDRProtocol: preregistracja, kontrole, werdykt

`Hypothesis` → `TIMDRProtocol.preregister()` zamraża parametry i liczy
odcisk SHA-256 z kanonicznego JSON (kolejność kluczy nie wpływa na odcisk).
`run_controls()` akceptuje wyłącznie jawnie dostarczony `ControlResult`
(brak kontroli = automatycznie nieprzeszedł). `run_test()` zwraca
`SUPPORTED`/`NOT_SUPPORTED`/`INCONCLUSIVE`:

- brak przekazanych kontroli, kontrole nieprzeszłe, lub brak `TestEvidence` → `INCONCLUSIVE`;
- kontrole przeszły i `p_value <= alpha` oraz `|effect_size| >= min_abs_effect_size` → `SUPPORTED`;
- kontrole przeszły, ale kryteria nie spełnione → `NOT_SUPPORTED`.

Żadna ścieżka nie pozwala warstwom modelu LTR ani modułom uczenia ustawić
tego werdyktu bezpośrednio — `TIMDR_AI_System.run()` zawsze przechodzi przez
`TIMDRProtocol`.

## FundamentalModelLTR: warstwy T/I/M/It/R/E

Sześć warstw (`LayerTTopology`, `LayerIInformation`, `LayerMModal`,
`LayerItTemporal`, `LayerRResonance`, `LayerEEmergence`), każda domyślnie
identycznością, każda wymienna przez argument `transform=` (jeden
callable jednoargumentowy). Przepływ danych `FundamentalModelLTR.forward()`:

1. `T` i `M` czytają surowe wejście **równolegle i bezpośrednio** — oba widzą to samo okno, nie łańcuch.
2. `I` scala ich dwie reprezentacje: `transform` dostaje jeden słownik `{"T":..,"M":..}`.
3. `It`, potem `R` przetwarzają dalej sekwencyjnie.
4. `E` na końcu łączy T/M/R: `transform` dostaje `{"T":..,"M":..,"R":..}`.

`forward()` zwraca pełny słownik stanu wszystkich warstw
(`{"T","M","I","It","R","E"}`), nie tylko wyjście `E`. Model z samymi
warstwami domyślnymi (identyczność) zwraca `{"T": raw, "M": raw, "R": raw}`
opakowane identycznością `E` — nie surowe wejście bez zmian.

```python
from timdr_ai_core import FundamentalModelLTR, LayerTTopology, LayerMModal

model = FundamentalModelLTR(
    topology=LayerTTopology(lambda w: ...),
    modal=LayerMModal(transform=lambda w: ...),
)
state = model.forward(window)  # {"T":..., "M":..., "I":..., "It":..., "R":..., "E":...}
```

### Operatory: realna matematyka do wpięcia

`timdr_operators.py` (extra `operators`, potrzebuje numpy) dostarcza gotowe,
wiernie przeniesione (1:1, z podanym źródłem) funkcje:

- `fft_dominant_mode(window, fs)` — ekstrakcja dominującego trybu (f, phi, A), port z `TIMDR-Modal-Formalism/timdr_modal/real_data_validation.py`; dla `LayerMModal`.
- `winding_number`/`crossing_number` — niezmienniki topologiczne krzywej osadzonej z opóźnieniem, port z `GIA-TIMDR/core/winding_crossing_ms_bridge.py`; dla `LayerTTopology`. Uczciwy zakres przeniesiony z repo źródłowego: odrzucone na danych syntetycznych, silny-lecz-częściowy sygnał na realnych łożyskach CWRU, niespójny na sejsmice/BTC — diagnostyczne, nie selekcyjne. `crossing_number` jest O(n²) pamięci i odmawia okien dłuższych niż `max_length` (domyślnie 3000, ~1,3 GB w najgorszym razie) z czytelnym `TimdrOperatorsError` zamiast próbować alokację i zawieść surowym `MemoryError` (realne okno Paderborn, 64000 próbek @ 64 kHz, wymagałoby ~65 GB) — dłuższe okna trzeba samodzielnie zdownsamplować lub podzielić.
- `Modality`/`is_resonant` — Aksjomat 5 gałęzi K, port z `TIMDR-Modal-Formalism/timdr_modal/phase_sync.py` (bez numpy); porównuje dwie modalności, nie pasuje do jednoargumentowego kształtu `LayerRResonance` — użyj jako osobne narzędzie porównawcze.

```powershell
.venv\Scripts\python.exe -m pip install -e ".[operators]"
python -m pytest -q tests/test_timdr_operators.py
python examples\run_timdr_operators_on_paderborn.py  # opcjonalnie: realne dane, patrz sekcja Paderborn
```

## Moduły uczenia: LearningSandbox, MLPClassifier

`learning_sandbox.py::LearningSandbox` — klasyfikator najbliższego centroidu:
`fit()` liczy centroidy klas z podziału `train`, wymaga też obecności
`calibration` i `holdout` (choć holdoutu nie dotyka), `predict_features()`
klasyfikuje deterministycznie (remis rozstrzyga niższy numer klasy).
`propose()` zwraca `CandidateProposal` z `requires_human_preregistration=True`
i `may_not_change_existing_preregistration=True`.

`neural_network.py::MLPClassifier` — generyczny, niezależny od TIMDR
jednowarstwowy MLP (ReLU + softmax, cross-entropy, pełny gradient descent,
ręcznie wyprowadzony backprop). Manualne gradienty zweryfikowane różnicami
skończonymi w `tests/test_neural_network.py` — to dowód poprawności
matematyki uczenia, nie użyteczności modelu na jakimkolwiek zbiorze danych.
Deterministyczny przy ustalonym `seed`.

Oba są prawdziwym uczeniem maszynowym i klasyfikacją — nie iluzją. Granica
jest architektoniczna, nie statystyczna: ich wynik (nawet wysoka dokładność)
nigdy nie wchodzi do `TIMDRProtocol.run_test()` automatycznie. Wymaga tego
osobna, nowa prerejestracja człowieka.

`b4_kitchen_learning_adapter.py` wymusza tę samą zasadę w drugą stronę:
`assess_imported_b4_result()` odrzuca gotowy, zaimportowany wynik ewaluacji
B4-Kitchen v0.3 jako dane treningowe — ewaluacja i trening to rozłączne
prerejestracje.

## PROTECT-90: uczenie badawcze na przebiegach EMT

Źródłem jest symulacja EMT (`TIMDR-Grid-Monitor`), nie pomiar polowej sieci
energetycznej. Żaden z poniższych baseline'ów nie jest algorytmem
zabezpieczenia ani wynikiem B4.

**Plan metadanych** — czteroklasowa klasyfikacja `sc_type` z 11 kolumn
metadanych scenariusza (`FEATURE_COLUMNS` w `protect90_learning_adapter.py`).
Przed pierwszym `fit()` zamraża się prerejestrację i podział stratyfikowany
(60/20/20, kolejność wg SHA-256, nie losowość runtime):

```powershell
.\run.bat --protect90-freeze "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
.\run.bat --protect90 "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
```

Druga komenda używa jedynie treningu i kalibracji; raport trafia do
ignorowanego przez Git `learning_runs/`.

**Plan waveform** — osobna hipoteza, 16 cech z przebiegów EMT (8 lokalizacji
× szczytowy prąd RMS / minimalne napięcie RMS, znormalizowane do pierwszych
200 ms; `sampling_rate_hz=6400`). Nie używa `sc_type`, lokalizacji
uszkodzenia ani czasu zdarzenia jako cech — ma własną prerejestrację
(`PROTECT90_WAVEFORM_MULTICLASS_v0.3`):

```powershell
.\run.bat --protect90-waveform-freeze "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"
.venv\Scripts\python.exe -m pip install -e ".[waveform]"
.\run.bat --protect90-waveform "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"       # nearest-centroid
.\run.bat --protect90-waveform-nn "C:\Users\jback\Downloads\a\TIMDR-Grid-Monitor"    # MLP (neural_network.py)
```

Hiperparametry MLP (`HIDDEN_UNITS=12`, `EPOCHS=800`, `LEARNING_RATE=0.05`,
`SEED=0`) są stałymi modułu w `protect90_waveform_nn_learning.py`, zamrożonymi
przed uruchomieniem na kalibracji. Dokładność na kalibracji z tymi
parametrami, jedno uruchomienie, ten sam podział: `0.6538` (MLP) vs `0.5288`
(najbliższy centroid), 104/104 próbek kalibracyjnych, holdout nietknięty.

## Granica odpowiedzialności

Status `SUPPORTED`/`NOT_SUPPORTED`/`INCONCLUSIVE` wynika WYŁĄCZNIE z
prerejestrowanego testu i kontroli dostarczonych przez
`TIMDRProtocol.run_test()`. Warstwy T/I/M/It/R/E przetwarzają reprezentacje
i same nie mogą ustanowić tego werdyktu. Moduły uczenia
(`LearningSandbox`, `MLPClassifier`) generują realne wyniki empiryczne
(dokładność klasyfikacji na kalibracji) — to nie jest fikcja — ale są
kandydatami badawczymi, izolowanymi od `TIMDRProtocol`: żaden nie wchodzi
do werdyktu bez osobnej, nowej prerejestracji człowieka.

## Ewidencja i graf pochodzenia (B4-Kitchen)

`evidence_runner.py` importuje i waliduje niemodyfikowalne raporty ewidencji
(np. `import_b4_kitchen_v03()`) — krzyżowo sprawdza deklarowany werdykt
przeciwko surowym p-value zamiast im ufać, i etykietuje wynik
`IMPORTED_CLAIM_NOT_INDEPENDENTLY_REEXECUTED`. `provenance_graph.py` buduje
graf DAG takich raportów i wykrywa cykle (algorytm Kahna).

```powershell
.\run.bat --graph          # buduje graf pochodzenia B4-Kitchen v0.3
.\run.bat --learn          # demo sandboxa uczenia bez dostępu do holdout
.\run.bat --b4-boundary    # pokazuje odrzucenie ewaluacji jako danych treningowych
```

## Lekki lokalnie, uczący się z sieci

Lokalny komputer wykonuje mało pracy: `LocalBudget` w `environment_policy.py`
ustawia jeden proces naraz (`max_parallel_jobs=1`) i limit 8 MB na epizod
(`max_episode_bytes`, egzekwowany np. przez `protect90_waveform_learning.py`
przed odczytem każdego pliku waveform). Rozwój źródłowy może zachodzić
automatycznie w sieci: cykl pobiera nowe dokumenty z zadeklarowanych
katalogów HTTPS (maks. 16 dokumentów na cykl, po maks. 2 MB) i aktualizuje
lokalny graf pochodzenia/słownik pojęć. Automatyczne uczenie sieciowe nie
pobiera ani nie uruchamia obcego kodu, nie wysyła danych lokalnych, nie
modyfikuje prerejestracji i nie ustanawia wyniku TIMDR.

```powershell
.\run.bat --online-learn "online_catalogs.json"    # skopiuj wcześniej online_catalogs.template.json
.\run.bat --online-rank                            # trasuje zebrane źródła po słowach kluczowych do 4 gałęzi TIMDR
.\run.bat --research-queue                         # buduje kolejkę kandydatów, nie tworzy hipotez ani mostów
```

Stan zapisuje się w ignorowanym przez Git `external_cache/`. Ranking i
kolejka badawcza to trasowanie/spis braków — nie stanowią wyniku badawczego
ani mostu między gałęziami.

Niezależnie od cyklu automatycznego, konkretny artefakt danych o znanym z
góry SHA-256 można pobrać jawnie przez `download_declared_source()`
(HTTPS-only, odmawia nadpisania istniejącego cache):

```powershell
.\run.bat --external-source "external_sources.json" "source-id"
```

Szablon: [external_sources.template.json](external_sources.template.json).
Plik trafia do `external_cache/`; dopiero nowa prerejestracja nadaje mu
status danych do analizy lub uczenia.

## Paderborn: zewnętrzna replikacja M/S

Trzy klasy: zdrowe `K001`, sztuczne EDM uszkodzenie pierścienia
zewnętrznego `KA01`, pierścienia wewnętrznego `KI01`. Archiwa `.rar` są
ignorowane przez Git. Zamrożenie selekcji nie czyta wartości sygnału —
tylko nazwy plików MATLAB w archiwach, podział 144/48/48:

```powershell
.\run.bat --paderborn-freeze
```

Prerejestracja hipotezy M/S (`PADERBORN_MS_HYPOTHESIS_v0.1`) definiuje kanał
`vibration_1`, cztery jednosekundowe okna na pomiar (`SAMPLE_RATE_HZ=64000`),
cztery cechy M/S, kontrolę dodatnią/ujemną, jednorazową ocenę holdoutu.
`paderborn_extractor.py` otwiera wyłącznie zamrożone pliki treningowe i
kalibracyjne — żądanie holdoutu odrzuca przed otwarciem archiwum:

```powershell
.\run.bat --paderborn-schema
```

## Testy

```powershell
.\run.bat --tests
```

lub ręcznie `python -m pytest -q` po `pip install -e ".[dev]"`. Testy
zależne od numpy używają `pytest.importorskip("numpy")` i pomijają się
bez niego. Testy wymagające realnych plików danych (waveform PROTECT-90,
archiwa Paderborn) nie są częścią tego zestawu — sprawdzają logikę
(podział, walidację prereg, odmowę holdoutu) na syntetycznych/tymczasowych
danych; uruchomienie na realnych danych wymaga zewnętrznego repo
`TIMDR-Grid-Monitor` (PROTECT-90) lub pobranych archiwów Paderborn, i jest
wywoływane ręcznie przez `run.bat`/`examples\*.py` jak wyżej.
