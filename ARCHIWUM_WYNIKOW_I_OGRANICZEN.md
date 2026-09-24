# Archiwum wyników i ograniczeń TIMDR-AI-Core

Ten dokument oddziela wyniki i ograniczenia od instrukcji uruchomienia. Status opisuje stan wykonanej pracy, nie rangę całej teorii.

## Potwierdzone technicznie

- Claim Graph zwraca odpowiedź wraz ze statusem i rekordami źródłowymi.
- `TIMDRProtocol` blokuje `SUPPORTED`, jeśli brakuje prerejestracji, kontroli lub wystarczającej ewidencji.
- Qwen 2.5 1.5B działa lokalnie na CPU.
- Pilot LoRA wykonał 12 kroków w 2 epokach, uczył 38 912 parametrów, zużył około 6,2 GB RAM i zapisał adapter.
- Qwen 3 4B GGUF został załadowany lokalnie na CPU bez AVX2.

## Wynik adaptera LoRA

Zamrożone porównanie na 22 pytaniach dało spadek średniej straty z **3,1367** do **3,1345**. Poprawa około **0,07%** potwierdza techniczne działanie adaptera, ale nie jest istotnym dowodem poprawy wiedzy modelu.

Pytania kontrolne nie mogą wejść do treningu. Raporty i adapter są lokalne w `learning_runs/`.

## Ograniczenia modeli lokalnych

- Adapter dla Qwen 1.5B nie pasuje do Qwen 3 4B.
- Qwen 1.5B potrzebuje około 6,3 GB RAM przy pierwszym pytaniu w oknie dialogowym.
- Niska strata nie mierzy prawdziwości, cytowania źródeł ani poprawnej odmowy poza zakresem.
- Model językowy nie może zmienić statusu Claim Graph.

## Zamknięte lub nierozstrzygające linie

### Diagnostyka aktywacji

- Warianty syntetyczne nie przeszły kontroli fałszywych alarmów albo wczesnego alarmu.
- Paderborn v0.7: `INCONCLUSIVE`, ponieważ kalibracja nie zawierała błędnych predykcji; holdout pozostał zamknięty.
- UCI HAR v0.9: nie potwierdzono przyrostu nad pewnością modelu.
- HARTH v1.1–v1.2: `INCONCLUSIVE`; nie wykazano przewagi rozszerzonych metryk nad bazą. Linię zamknięto bez dalszego strojenia.

Szczegóły są w `prereg/ACTIVATION_PROBE_*.md`.

### Mosty TIMDR

- `MC_K↔G` jest **odrzucony**: wynik był artefaktem gęstości kratownicy, a nie wiarygodnym sygnałem.
- `MC_M/S↔G` jest **potwierdzoną diagnostyką**, nie selektorem: silny wynik na łożyskach, częściowy na sejsmice i brak na BTC.
- Most Fouriera jest ograniczony do idealnego pojedynczego modu gaussowskiego; nie jest ogólnym testem dla dowolnego sygnału realnego.

Źródła: [wynik mostów Möbiusa](../GIA-TIMDR/docs/geometry/RESULT_MOBIUS_COHERENCE_BRIDGES_REAL_DATA.md) oraz [specyfikacja gałęzi](../GIA-TIMDR/docs/theory/TIMDR_Branch_Specification.md).

## Dalsza praca

Następny krok dla adaptera to większy, zatwierdzony zbiór treningowy oraz ponowne porównanie wyłącznie na tym samym zamrożonym holdoucie. Nie należy stroić danych ani progów po wyniku kontroli.
