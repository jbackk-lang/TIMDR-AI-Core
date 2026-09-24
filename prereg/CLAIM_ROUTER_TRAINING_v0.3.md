# Claim Router v0.3

v0.3 jest nową wersją po dwóch nieudanych kalibracjach, bez otwarcia holdoutu.
Używa podobieństwa znakowych n-gramów zamiast licznika słów, dzięki czemu
odmiany polskie i skróty są mniej kruche. `holdou*` normalizuje się do
`holdout`. Werdykt nadal pochodzi wyłącznie z Claim Graph.

Nowa calibration i holdout zostały zapisane przed treningiem. Holdout wolno
otworzyć tylko przy 100% trafności i zgodności logicznej calibration.
