# Faza 2 — kontroler inference TIMDR (nieuruchomiony plan)

Cel: osobny kontroler diagnostyczny dla odpowiedzi Qwena po zakończeniu
adaptacji dokumentowej. Nie ma zmieniać odpowiedzi ani wag modelu w locie.

## Minimalny przepływ

`prompt → Qwen + adapter → odpowiedź i stany ukryte → adapter TIMDR → raport`.

Kontroler zwraca wyłącznie: metryki, status `OK / REVIEW / INCONCLUSIVE`,
hash konfiguracji oraz powody flag. Nie wybiera faktów, nie nadpisuje tekstu,
nie aktualizuje toru odniesienia i nie uczy modelu podczas inference.

## Warunki przed implementacją

1. Osobny zbiór kalibracyjny reprezentacji, zamrożony przed progiem.
2. Jawne definicje kanałów i rozłączne źródła: energia dla rho, kierunek dla J.
3. Kontrola dodatnia (wstrzyknięta degradacja wejścia) oraz ujemna (szum i
   parafraza bez zmiany sensu).
4. Holdout pytań i faktów, niedostępny podczas projektowania progu.
5. Porównanie z prostym baseline'em: pewność/entropia następnego tokenu.

Bez pięciu warunków faza 2 pozostaje projektem inżynierskim, a nie dowodem,
że kontroler poprawia wiarygodność odpowiedzi.
