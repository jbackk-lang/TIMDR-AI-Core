# Qwen bazowy vs Qwen + Capsule — plan zamrożony v0.1

**Status:** zamrożony przed pierwszym przebiegiem.

## Pytanie

Czy niezmieniony lokalny `Qwen2.5-1.5B-Instruct`, po otrzymaniu wyłącznie
deterministycznie dobranych rekordów Fazy 1C, lepiej zachowuje ich źródła,
statusy i ograniczenia niż ten sam model bez rekordów?

To nie jest test jakości ogólnej modelu, treningu, LoRA ani empiryczny wynik
TIMDR. Testuje wierność ścieżce dowodowej.

## Zamrożone wejścia

- model: `Qwen/Qwen2.5-1.5B-Instruct`, rewizja
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
- kapsuła: `data/timdr_knowledge_injection_v0.2.json`, SHA-256
  `feca930054b9827b06cd1f31ac72f8ac20f759c3795fd9f88ec69b3fd7250327`;
- 11 pytań z `data/qwen_capsule_eval_v0.1.json`: siedem o pokrytym
  dowodzie i cztery kontrole negatywne;
- dekodowanie deterministyczne, `max_new_tokens=72`, bez LoRA;
- CPU: dwa wątki. To ogranicza zasoby, nie zmienia merytorycznej reguły.

## Porównanie

Wariant `base` dostaje wyłącznie to samo pytanie i instrukcję formatu.
Wariant `capsule` dostaje tę samą instrukcję oraz rekordy zwrócone przez
`phase1c_knowledge_injection.py`. Żaden wariant nie ma dostępu do sieci,
uczenia ani zmiany wag.

Model otrzymuje polecenie cytowania identyfikatorów rekordów wyłącznie wtedy,
gdy są dostarczone. Brak rekordu ma kończyć się dokładną etykietą
`INCONCLUSIVE_NO_MATCH`.

## Ocena

Automat sprawdza wyłącznie zgodność hashy, deterministyczną ścieżkę dowodową,
obecność właściwych identyfikatorów w cytowaniu i odmowę w kontrolach
negatywnych. Nie może uczciwie ocenić prawdziwości odpowiedzi językowej przez
proste dopasowanie słów.

Przebieg tworzy zatem kolejkę ręcznej oceny z kryteriami: zgodność merytoryczna,
zachowanie statusu i ograniczenia, cytowanie oraz odmowa poza zakresem. Kolejka
jest celowo pozbawiona nazwy wariantu.

## Blokada

Nie zmieniać pytań, selektora, kapsuły, modelu ani dekodowania po pierwszym
przebiegu. Awaria techniczna daje `INCONCLUSIVE_TECHNICAL`; poprawiony przebieg
na zmienionych wejściach wymaga nowej wersji planu. Faza 2 pozostaje zamknięta
niezależnie od wyniku tego testu.
