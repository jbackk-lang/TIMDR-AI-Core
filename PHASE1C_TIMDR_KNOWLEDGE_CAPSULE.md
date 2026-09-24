# Faza 1C — kapsuła wiedzy TIMDR z zakresem i ograniczeniami

**Status:** gotowa do technicznej walidacji; nie jest treningiem, LoRA ani
kontrolerem odpowiedzi modelu. Wersja 1B pozostaje nienaruszona jako mniejsza,
zamrożona kapsuła. Wersja 1C jest nowym, oddzielnym artefaktem.

## Cel

Kapsuła v0.2 przygotowuje odtwarzalny kontekst dla późniejszego testu modelu
językowego. Każdy rekord posiada obiekt, operator, twierdzenie, status, zakres,
ograniczenie oraz dokument źródłowy z SHA-256.

`pytanie -> jawne mapowanie operatora -> rekordy dowodowe -> weryfikacja hashy -> ograniczony kontekst`

Jeśli pytanie nie pasuje do jawnej mapy, system zwraca
`INCONCLUSIVE_NO_MATCH`. Jeżeli dokument źródłowy zmienił się od zamrożenia,
wynik brzmi `INCONCLUSIVE_SOURCE_CHANGED`.

## Co rozszerza 1C

- rozróżnia zamrożony tor GIA od wyniku operatora;
- rozróżnia ograniczony most Fouriera od ogólnego testu sygnałów;
- rozróżnia odrzucony `MC_K<->G` od domenowo kwalifikowanego,
  diagnostycznego `MC_M/S<->G`;
- rozróżnia definicję źródeł `rho` i `J` od nieudowodnionej niezależności;
- chroni zasadę kalibracji i zamrożenia przed tuningiem po holdoucie.

## Walidacja techniczna

```powershell
python test_phase1c_knowledge_injection.py -v
python examples/phase1c_knowledge_injection.py "Czy most Fouriera jest ogólnym testem dla każdego realnego sygnału?"
python examples/phase1c_knowledge_injection.py "Jak upiec chleb na zakwasie?"
```

Test obejmuje zgodność hashy, kompletność statusu/zakresu/ograniczeń,
kierowanie pytań holdoutu, kontrolę negatywną i rozdzielenie mostu odrzuconego
od diagnostycznego.

## Następna bramka

Potrzebny jest osobny, pre-rejestrowany test porównawczy modelu bazowego i
modelu z kapsułą: ten sam limit tokenów i dekodowanie, ocena osobno dla
cytowania źródeł, wierności statusowi/ograniczeniom, odmowy poza zakresem i
jakości tekstu. Pytania oceny nie mogą służyć do zmiany selektora ani kapsuły.

Faza 2 — kontroler inferencji — pozostaje zablokowana, dopóki kapsuła nie
wykaże na niezależnym holdoucie poprawy wierności bez maskowania braków
dowodowych.
