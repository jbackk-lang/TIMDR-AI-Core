# TIMDR-AI-Core

Lokalny rdzeń do pracy z wiedzą i eksperymentami TIMDR. Łączy trzy rzeczy:

- **Claim Graph**: ustala status odpowiedzi i jej źródła;
- **protokół TIMDR**: nie pozwala ogłosić wyniku `SUPPORTED` bez prerejestracji, kontroli i ewidencji;
- **lokalne uczenie**: małe, odtwarzalne eksperymenty na CPU.

## Uruchomienie okna dialogowego

Kliknij [`run_answer_engine.bat`](run_answer_engine.bat). Okno najpierw sprawdza pytanie przez Claim Graph, a następnie może dodać krótkie wyjaśnienie lokalnego Qwen 1.5B z adapterem LoRA. Status, źródła i ograniczenia zawsze pochodzą z Claim Graph.

Pierwsze pytanie ładuje model do pamięci. Na tym komputerze potrzeba około 6,3 GB RAM i dwóch wątków CPU.

## Co jest gotowe

- Deterministyczna bramka Claim Graph dla odpowiedzi ze źródłami.
- `TIMDRProtocol`, który rozdziela propozycję modelu od werdyktu badawczego.
- Moduły uczenia na CPU: klasyfikator centroidowy, mały MLP i ograniczone adaptery badawcze.
- Qwen 2.5 1.5B z lokalnym adapterem LoRA.
- Qwen 3 4B GGUF uruchomiony lokalnie na CPU bez AVX2.
- Zamrożony zestaw 22 pytań kontrolnych: [`data/timdr_text_holdout_v0.1.json`](data/timdr_text_holdout_v0.1.json).

## Stan adaptera LoRA

Pilot na sześciu zatwierdzonych parach pytanie–odpowiedź wykonał 12 kroków na CPU i zapisał adapter. Porównanie na 22 pytaniach niewidzianych w treningu dało niewielką poprawę średniej straty: **3,1367 → 3,1345** (około **0,07%**). Adapter jest warstwą wyjaśniającą, ale nie zastępuje Claim Graph.

## Najważniejsze polecenia

```powershell
.\run.bat --tests
.\run.bat --graph
.\run.bat --learn
.\run.bat --online-rank
```

Dokumentacja modeli: [Qwen 1.5B](GENERATIVE_MODEL_SETUP.md) i [pilot LoRA](TIMDR_LORA_CPU_PILOT.md).

## Dokumentacja i archiwum

- [Archiwum wyników, ograniczeń i zamkniętych prób](ARCHIWUM_WYNIKOW_I_OGRANICZEN.md)
- [Kapsuła wiedzy TIMDR](PHASE1C_TIMDR_KNOWLEDGE_CAPSULE.md)
- [Reguła kalibracji i zamrożenia](/github.com/jbackk-lang/GIA-TIMDR/docs/theory/TIMDR_CALIBRATION_FREEZE_RULE.md)
- [Specyfikacja gałęzi](/github.com/jbackk-lang/GIA-TIMDR/docs/theory/TIMDR_Branch_Specification.md)

Wyniki eksperymentów są kandydatami badawczymi. Werdykt empiryczny wymaga osobnej prerejestracji i kontroli.
