# Faza 1B — TIMDR Knowledge Injection

Status: przygotowanie źródeł i testów. Faza 2 pozostaje zablokowana.

## Znaczenie operacyjne

„Knowledge Injection” oznacza tu obserwowalny, odtwarzalny kontekst dowodowy,
nie zmianę wag, LoRA ani ukryty prompt. Każde zapytanie tworzy trajektorię:

`zapytanie → selektor operatorów → rekordy dowodowe → hashe źródeł → kontekst odpowiedzi`.

Model językowy może później otrzymać wyłącznie taki kontekst; odpowiedź bez
rekordu dowodowego ma status `INCONCLUSIVE`, nie jest uzupełniana z pamięci.

## Zamrożone elementy v0.1

- Kapsuła: `data/timdr_knowledge_injection_v0.1.json`.
- Operatory i modalności: M/S/sygnał, G/geometria, K/modalność,
  META-DYNAMICS/dynamika agregatowa, Chronoproces/czas, Protocol/kontrola.
- Tor: deterministyczna selekcja słów kluczowych, maksymalnie trzy rekordy,
  wraz z SHA-256 wskazanych dokumentów.
- Baseline: Qwen bez LoRA, bez kapsuły.
- Warunek: Qwen z kapsułą otrzymuje identyczny limit tokenów i decoding jak baseline.
- Holdout: pięć pytań w polu `holdout`, nieużywanych do wyboru rekordu ani progu.

## Kryteria przed uruchomieniem porównania Qwen

1. Każde pytanie holdoutu wybiera wymagane rekordy.
2. Każdy rekord wskazuje istniejący dokument i hash.
3. Zakłócenia tekstowe nie zwracają dowodów przypadkowo.
4. Progi lub selektor nie są zmieniane po odpowiedziach baseline'u.
5. Wynik porównania jest osobno zapisywany dla: trafności cytowania,
   zgodności z wymaganymi terminami, odmowy przy braku dowodu i jakości tekstu.

## Granica

To nie udowadnia, że TIMDR poprawia zdolności modelu. Sprawdza wyłącznie, czy
zamrożona ścieżka dowodowa ogranicza konfabulację lepiej niż model bazowy.
Faza 2 — kontroler inference — może zostać otwarta wyłącznie po przejściu
tego porównania z osobnym holdoutem i kontrolami dodatnią/ujemną.
