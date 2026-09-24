# TIMDR Claim Graph / Logic Gate v0.1

To jest warstwa deterministyczna inspirowana architekturą potoku
`math-validator-3.0`: jawne wejścia, niezależne reguły, kontrola sprzeczności
i bezpieczny wynik zamiast domyślnego zgadywania.

Nie kopiuje jednak filtra topologii równań. Dziedzina ciągłości `1/x` nie jest
modelem prawdziwości zdania tekstowego. Przenośna jest **dyscyplina bramki**,
nie metafora.

## Przepływ

`pytanie -> węzeł grafu twierdzeń -> werdykt + wymagania + zakazy -> renderer`

Opcjonalny model językowy może jedynie sparafrazować wynik renderera. Funkcja
`gate_candidate()` odrzuci jego tekst, gdy zabraknie statusu lub cytatu albo
pojawi się stwierdzenie zakazane przez rekord. Wtedy interfejs ma użyć
deterministycznego renderera, nie ponawiać losowego generowania.

## Przykład

```powershell
python examples\claim_graph_demo.py "Czy MC K-G jest potwierdzony?"
python test_claim_graph_gate.py -v
```

## Gotowy lokalny interfejs

Uruchom `run_answer_engine.bat` albo:

```powershell
python timdr_answer_ui.py
```

Okno nie ładuje Qwen i nie wymaga internetu. Jest to celowe: odpowiedź pochodzi
z grafu twierdzeń, a nie z predykcji następnego tokenu.

## Zakres

Graf v0.1 obejmuje osiem dokładnie zapisanych twierdzeń TIMDR. To nie jest
ogólny silnik rozumowania ani dowód wszystkich twierdzeń w repo. Każde nowe
twierdzenie wymaga osobnego rekordu, źródła, statusu, zakresu i testu
sprzeczności.
