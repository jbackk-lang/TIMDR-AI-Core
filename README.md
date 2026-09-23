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

### Eksploracyjna diagnostyka aktywacji jednej warstwy

`activation_diagnostics.py` mierzy wyłącznie kolejne aktywacje **tej samej**
warstwy zamrożonego modelu: `delta_time` (RMS różnicy między krokami),
`lambda_channel` (dyspersja między kanałami w jednym kroku) i
`tau_lambda_time` (tempo zmiany tej dyspersji). To lokalne definicje
inspirowane TIMDR, nie utożsamione z operatorami innych domen. Nie ma tu
odejmowania wektorów z różnych warstw ani twierdzenia o „czasie myślenia”.

`fit_healthy_reference()` używa wyłącznie poprawnych trajektorii
kalibracyjnych i zamraża centrum, skalę oraz próg alarmu. Wynik demonstracji
można zobaczyć poleceniem:

```powershell
.\.venv\Scripts\python.exe examples\activation_probe_demo.py
```

Demonstracja używa małego MLP i **wyłącznie syntetycznych** sekwencji.
Kolejne kroki są tu kolejnymi wejściami do tej samej sieci, a nie
tokenami generowanymi przez model językowy; nie jest to test halucynacji.
W pierwszym, niepoprawianym po obejrzeniu wyniku przebiegu: 74/200
końcowych predykcji było błędnych; AUC diagnostyki aktywacji wyniosło
około 1,00 wobec 0,60 dla `1 - max(softmax)`. Jednak alarm wystąpił też
u 15/26 (57,7%) zakłóconych, lecz nadal poprawnie sklasyfikowanych
przypadków. Ta kontrola fałszywych alarmów **nie przechodzi** — bardzo
wysokie AUC nie uprawnia do ogłoszenia działającego detektora błędów.
Zakłócenie jest osobną cechą scenariusza, a etykieta błędu wynika z
rzeczywistej poprawności predykcji, nie z obecności zakłócenia.

Kod nie produkuje werdyktu `TIMDRProtocol` i nie dotyka istniejących
holdoutów. Zanim padnie twierdzenie o dodatkowej informacji względem
confidence, potrzebny jest osobny zamrożony eksperyment na danych
rzeczywistych, z kontrolą trudnych poprawnych przykładów i porównaniem
modelu łączącego oba kanały z modelem używającym samej pewności.

**Wariant v0.2 (osobny, eksploracyjny):** `activation_diagnostics_v02.py`
rozdziela podpisaną średnią zmianę kanałów od rozrzutu tej zmiany;
`lambda` opisuje rozkład aktywności między kanałami przez znormalizowaną
entropię, a `tau` mierzy tempo jej zmiany względem wcześniejszej
aktywności tej samej sekwencji. Drugi wariant referencji dzieli zdrowe
przykłady na łatwiejsze/trudniejsze według ruchu **wejścia przed ostatnim
krokiem** — bez użycia prawdziwej klasy, poprawności wyniku ani końcowego
zakłócenia. Stare definicje i wynik v0.1 pozostały bez zmian.

```powershell
.\.venv\Scripts\python.exe examples\activation_probe_v02.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_activation_diagnostics*.py
```

Na nowym syntetycznym zestawie (400 przypadków, inne ziarna niż v0.1)
porównanie na **tych samych** przykładach dało:

| Wariant | AUC błędu | Fałszywy alarm: trudne poprawne | Czułość: zakłócone błędne |
| --- | ---: | ---: | ---: |
| v0.1, stara metryka | 0,874 | 25,0% | 60,5% |
| v0.2, nowe cechy bez podziału | 0,942 | 77,1% | 98,7% |
| v0.2, podział według trudności | 0,931 | 70,8% | 92,1% |
| Sama niepewność `1 - max(softmax)` | 0,584 | nie ustalono wspólnego progu | nie ustalono wspólnego progu |

Podział według trudności ograniczył część fałszywych alarmów v0.2, ale
**nie uratował kontroli negatywnej**. Większe AUC i czułość są tu okupione
zbyt wieloma alarmami dla poprawnych wejść. Nie dostrajano progu po tym
wyniku. „Wczesny alarm” również nie został przetestowany: zakłócenie
w tym generatorze pojawia się dopiero w ostatnim kroku, więc nie ma
wcześniejszego momentu, w którym taki alarm mógłby się pojawić.

**Wariant v0.3 — kierunek przy podobnym Δ i wczesny alarm:** osobny plan
przed pierwszym przebiegiem jest w
`prereg/ACTIVATION_PROBE_v0.3_SYNTHETIC_PLAN.md` (SHA-256 przed przebiegiem:
`c3695768d418ad5046fc4d8856e2280137738a35bf51ea88e9ce30677c03a9a1`).
Ponieważ plan pozostaje lokalnym, niezatwierdzonym commitem, hash nie jest
niezależnym dowodem prerejestracji. Test dotyczy nadal jednego zamrożonego
MLP i **syntetycznych** trajektorii, nie modeli językowych ani danych
rzeczywistych. Typ wejścia jest tylko kontrolą raportową; prawdziwa klasa
nie wchodzi do sygnału alarmowego.

```powershell
.\.venv\Scripts\python.exe examples\activation_probe_v03.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_activation_diagnostics_v03.py
```

W pierwszym przebiegu części A: 667 zakłóconych-błędnych i 133
zakłóconych-poprawnych. Tylko **najwyższy** z pięciu przedziałów Δ miał
co najmniej 10 przykładów każdej grupy (658/91); pozostałe cztery nie
kwalifikowały się. W tym jednym przedziale `shape_js` osiągnęło AUC
`0,933`, podpisany kierunek KS `0,583`, a `locality` AUC `0,185` w
prerejestrowanym kierunku. Formalny warunek *kandydata do replikacji*
przeszedł (91 par), ale to **nie potwierdza** samodzielnej informacji
o kierunku/kształcie: bardzo szeroki górny przedział nadal może mieszać
różne wielkości Δ. Wymagana jest nowa, węziej dopasowana próba v0.4.

W części B Λ zaalarmowała przed błędem tylko w `10/386` kwalifikujących
się sekwencji (lead-recall `2,6%`); fałszywy alarm wystąpił w `7/214`
bezbłędnych (`3,3%`). Confidence wyprzedziła `251/386` błędów (`65,0%`),
ale alarmowała też w `113/214` bezbłędnych (`52,8%`). Ani samo Λ, ani
baseline nie daje tu użytecznego, dobrze skalibrowanego wczesnego alarmu;
bramka v0.3 dla Λ **nie przeszła**. Progów nie poprawiano po wyniku.

**Wariant v0.4 — pięć dopasowanych poziomów Δ:** plan zapisano przed
pierwszym przebiegiem w `prereg/ACTIVATION_PROBE_v0.4_SYNTHETIC_PLAN.md`;
jego SHA-256 przed testem to
`db98b5b09de2d3ae73e46e2de1b559d68baf73e4a4dd3fa73fd9ce3282c047ab`.
Jest to lokalny zapis, nie niezależnie poświadczona prerejestracja.
Kod nie zmienia v0.3. Uruchomienie:

```powershell
.\.venv\Scripts\python.exe examples\activation_probe_v04.py
```

W syntetycznym teście A było po 300 przypadków na każdy docelowy poziom
Δ (1,5; 2,0; 2,5; 3,0; 4,0), bez wykluczeń technicznych. Każdy poziom
zawierał ≥20 błędnych i ≥20 poprawnych predykcji. AUC `shape_js`
wyniosło odpowiednio **0,816; 0,840; 0,933; 0,926; 0,944**, a AUC
samej wielkości Δ **0,576; 0,531; 0,452; 0,477; 0,560**. Wszystkie
trzy typy wejścia (pierwsza przewidziana klasa) również spełniły
zamrożone kryterium: AUC `shape_js` 0,937–0,953. Zatem A przechodzi
**bramkę kandydata do replikacji**, lecz nie jest wynikiem na danych
rzeczywistych ani dowodem uniwersalnego operatora TIMDR. `shape_js`
mierzy kształt zmiany **symetrycznie**, nie jej kierunek. Na poziomie
Δ=1,5 prosty baseline confidence miał wyższe AUC (0,890 vs 0,816),
więc nie ma podstaw twierdzić, że `shape_js` zawsze wnosi przewagę nad
pewnością modelu.

W B zastosowano **ten sam narastający generator i te same progi co v0.3**,
tylko nowy seed. Λ wyprzedziła 6/390 błędów (1,5%), z 10/210 (4,8%)
fałszywych alarmów; confidence wyprzedziła 254/390 (65,1%), ale dała
101/210 (48,1%) fałszywych alarmów. Bramka wczesnego alarmu Λ ponownie
**nie przeszła**. Następny uczciwy krok dla A to niezależny, rzeczywisty
zbiór z dostępem do aktywacji i porównanie z confidence przy zamrożonym
protokole; nie dostrajanie tego testu syntetycznego.

**Paderborn v0.5–v0.7 — próba przejścia na rzeczywiste drgania:**
v0.5 i v0.6 zatrzymały się na technicznym schemacie długości pomiaru;
każda zmiana ma osobny plan i notatkę o przyczynie. W v0.7 odczyt
train/calibration zadziałał, ale bramka mocy dała **0 błędów na 336
przejściach kalibracyjnych** (model poprawny we wszystkich). Dlatego
wynik diagnostyki aktywacji jest `INCONCLUSIVE` i holdout pozostaje
zamknięty. Nie osłabiamy modelu po wyniku po to, by wytworzyć błędy.
Szczegóły: `prereg/ACTIVATION_PROBE_v0.7_CALIBRATION_RESULT.md`.

**UCI HAR v0.9 — niezależne rzeczywiste pomiary, aktywacje tej samej
warstwy:** zamiast osłabiać model Paderborn użyto danych czujników
telefonu od innych osób. Plan v0.8 nie ruszył z przyczyn technicznych
(Windows Device Guard zablokował `scikit-learn` przed odczytem danych);
v0.9 zamroził model NumPy 561→64→6. Odczytano wyłącznie oficjalne
`train/`, rozdzielone po osobach na fit i calibration; `test/` pozostał
nieotwarty. Bramka liczebności przeszła (193 błędne i 1442 poprawne
pary u 5 osób), ale `shape_js` miał AUC **0,486**, wobec **0,709** dla
niepewności modelu. To negatywna wskazówka kalibracyjna, nie wynik
potwierdzający i nie powód do dostrajania po fakcie. Plan, kod i pełne
liczby: `prereg/ACTIVATION_PROBE_v0.9_UCI_HAR_PLAN.md`,
`activation_probe_uci_har.py`,
`prereg/ACTIVATION_PROBE_v0.9_UCI_HAR_CALIBRATION_RESULT.md`.

**HARTH v1.1–v1.2 — linia zamknięta, bez potwierdzenia przyrostu:**
Na rzeczywistym zbiorze HARTH jednorazowy test pięciu odłożonych osób
porównał diagnostykę błędu opartą na niepewności modelu i średniej
wielkości zmiany aktywacji z wariantem dodającym `Δ-trajectory` oraz
`Λ-instability`. Dla MLP AUC wyniosło **0,9211** (baza) i **0,9176**
(rozszerzenie); różnica **−0,0034**, 95% przedział bootstrapu po osobach
**[−0,0143; +0,0095]**. Zgodnie z regułą testu werdykt to
**INCONCLUSIVE**: nie wykazano przewagi ani wiarygodnego pogorszenia.
Późniejsze, wyłącznie eksploracyjne porównanie na dwóch osobach
kalibracyjnych także nie wykazało przyrostu: różnica AUC wyniosła
**−0,0048** dla MLP i **−0,0104** dla lekkiego modelu rekurencyjnego
z pamięcią (reservoir). Tych dwóch osób nie traktujemy jako nowego
niezależnego holdoutu.

`shape_js` nie było testowane na HARTH; jego wcześniejszy wynik dotyczy
UCI HAR v0.9. Na HARTH nie testowano też wyprzedzania błędów w czasie.
Wniosek jest wąski: **w badanym ustawieniu nie potwierdzono dodatkowej
wartości tych dwóch metryk ponad przyjętą bazę**. Nie dowodzi to braku
jakiejkolwiek struktury diagnostycznej aktywacji ani nieskuteczności
wszystkich architektur. Linię HARTH zamknięto bez dalszego strojenia;
surowe archiwum i lokalne raporty usunięto na życzenie użytkownika.

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
