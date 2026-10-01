# Dostosowanie rdzenia do drogowskazów TIMDR

Stan: 1 października 2026. To poprawa przepływu i kontroli, nie dowód poprawy wiedzy sieci.

## Dwie drogi

**Sygnał:** typ sygnału → pole → rezonans → sito → samokorekta → geometria. Każdy aktywny etap wymaga rzeczywistego adaptera domenowego. Etap opcjonalny można pominąć tylko z zapisaną przyczyną. Plan zawiera zegar, kotwicę i osobną referencję kalibracyjną. Wszystkie etapy mają dostęp do kopii pierwotnych danych; ich wyniki nie zastępują bezpowrotnie surowego sygnału. Model nie dopasowuje referencji na ocenianych danych.

**Tekst:** pytanie/twierdzenia → źródła → kandydat modelu → kontrola publikacji. Okno odpowiedzi rzeczywiście używa tej drogi. Źródła i status nadal pochodzą z Claim Graph. Tekst nie jest utożsamiany z modalnością fizyczną.

Wynik operatorów jest kandydatem. Osobno wykonuje się test według zamrożonej hipotezy. Drogowskazy nie wymagają sztucznego dodawania geometrii do zadania bez danych przestrzennych.

## Co zmieniono

- `timdr_workflow.py`: jawne etapy, wymagane adaptery, oddzielne ścieżki i zapis wykonania/pominięcia.
- `TIMDR_AI_System(workflow=...)`: możliwość użycia tego przepływu zamiast dawnych adapterów LTR. Stare warstwy pozostają dostępne, ale ich domyślne funkcje nadal tylko przekazują dane.
- `TIMDRProtocol.run_test(..., preregistration=plan)`: wymaga planu, zgodnej metody i wyników przypisanych do jego fingerprintu. Zmiana hipotezy lub kryteriów jest odrzucana. Domyślny minimalny efekt wynosi teraz 0,3, zgodnie z `protocol_config.json`; dziedzina może jawnie wybrać inne kryterium przed oceną.
- W `TIMDR_AI_System.run` nie tworzy się już prerejestracji na podstawie wywołania zawierającego wyniki. Plan należy przygotować wcześniej.
- Nieskończone efekty oraz NaN są blokowane.
- Swobodne wyjaśnienie modelu jest odrzucane. Sprawdzenie słów, cytowań i statusu nie wystarcza do sprawdzenia znaczenia. Do czasu walidacji parafraz publikowana jest odpowiedź źródłowa; `explanation_status` pokazuje, co stało się z kandydatem.
- Poprawiono normalizację odmiany słowa „holdout” oraz zbyt szerokie dopasowanie pytań o Chronoproces.

## Przykład API protokołu

```python
from timdr_ai_core import TIMDRProtocol, Hypothesis, ControlResult, TestEvidence

protocol = TIMDRProtocol()
# Przed testem: rzeczywisty plan powinien wskazywać również dane, podział,
# kontrolki, kierunek hipotezy i regułę porównania z mocnym baseline'em.
plan = protocol.preregister(Hypothesis(
    "demo", "Tylko przykład API", "Zadany efekt",
    {"method": "synthetic demonstration test"}))
# Poniższe liczby są sztuczne, nie stanowią wyniku empirycznego.
evidence = TestEvidence(.01, .45, "synthetic demonstration test",
                        preregistration_fingerprint=plan.fingerprint)
result = protocol.run_test(ControlResult(True, True), evidence,
                           preregistration=plan)
```

Gotowy przykład: `python examples/protocol_demo.py`.

## Co sprawdzają testy

Brak planu, niepowiązane wyniki, zmiana referencji/metody, nieprawidłowe liczby, kolejność etapów, izolacja surowych danych, brak ukrytych funkcji tożsamościowych i próby dopisania niezweryfikowanych twierdzeń. Kontrola sygnałowa używa rzeczywistego operatora FFT: ton 40 Hz zgodny z zamrożoną kotwicą przechodzi, ton 80 Hz nie przechodzi. Jest to kontrola mechaniki na syntetycznym sygnale.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests test_claim_graph_gate.py test_claim_router_learning.py test_claim_router_learning_v02.py test_claim_router_learning_v03.py test_claim_router_v04.py test_phase1b_knowledge_injection.py test_phase1c_knowledge_injection.py -p no:cacheprovider
```

Sprawdzenie po zmianach: **103 testy przeszły** (testy rdzenia oraz wybrane testy Claim Graph, routerów i kapsuł wiedzy). Dwa ostrzeżenia pytest dotyczą klasy danych `TestEvidence`, której pytest nie traktuje jako klasy testowej. Testy odrzucania błędnych odpowiedzi używają podstawionych kandydatów; nie są nowym benchmarkiem Qwen.

## Granice

Fingerprint sprawdza powiązanie i integralność, nie dowodzi daty prerejestracji. Rzeczywisty eksperyment wymaga zewnętrznego zapisu planu przed oceną danych, np. historii Git, oraz runnera odtwarzającego kontrole i metryki. Wyniki i kontrolki dostarczone przez wywołującego nie stają się prawdziwe przez samą bramkę. Ta zmiana nie egzekwuje jeszcze wszystkich dziedzinowych reguł mocy, wielokrotnych porównań i kierunku hipotezy.

Controller nie implementuje całego formalizmu M/S, G, K i META ani nowego sita dla każdej domeny. Wymusza jawną konstrukcję adapterów. Nie wykazano poprawy odpowiedzi Qwen lub accuracy klasyfikatora i nie przeprowadzono nowego treningu. Nie ma podstaw do zmiany architektury sieci tylko z powodu tych usterek.

Źródła lokalne: GIA-TIMDR/docs/DROGOWSKAZY_TIMDR.md, docs/SKILL_timdr-signal-framework.md i docs/theory/TIMDR_Chronoprocess.md. Nowsze drogowskazy mają więcej informacji niż wcześniejszy zestaw samych nazw warstw T/I/M/It/R/E.
