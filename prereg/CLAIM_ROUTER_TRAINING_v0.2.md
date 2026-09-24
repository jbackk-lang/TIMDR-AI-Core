# Claim Router v0.2 — nowa kalibracja

Wersja v0.1 ujawniła dwa błędy calibration bez otwarcia holdoutu. v0.2 ma
osobny plan danych, normalizację form `holdout*`, więcej negatywów i z góry
ustawiony próg niepewności 1.0. Poprzednia calibration jest teraz częścią
rozwoju; nowa calibration jest rozłączna. Holdout pozostaje zamknięty.

Brama wejścia do holdoutu jest ścisła: 100% trafności routera i 100% zgodności
z Claim Graph na nowej calibration. Nawet przy przejściu bramki graf nadal
pozostaje jedynym źródłem werdyktu.
