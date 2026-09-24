# Qwen LoRA source-corpus — plan oceny po treningu v0.1

Status: plan zapisany w trakcie treningu; **nie zmienia bieżącego adaptera**.

## Zamrożone wejścia

- Korpus: `data/timdr_source_lora_v0.1.json`.
- Ocena językowa: dokładnie 11 fragmentów z pola `evaluation` tego pliku.
- Źródła oceny: `TIMDR_CALIBRATION_FREEZE_RULE.md` oraz
  `TIMDR_Geometry_From_EventGraph.md`.
- Ich identyfikatory i SHA-256 muszą zgadzać się z `plan.json` uruchomienia.
- Zakłócenia: `data/timdr_lora_perturbation_v0.1.json`.

Nie wolno dodawać fragmentów, zmieniać długości kontekstu, liczby generowanych
tokenów ani wag adaptera po odczycie wyniku.

## Pomiary

1. **Loss odłożonych fragmentów** przed/po adapterze.
2. **Drift reprezentacji**: względna norma L2 i cosinus ostatnich stanów
   ukrytych modelu bazowego oraz modelu z adapterem.
3. **Kanał rho-adapter**: bezwzględna różnica energii stanów ukrytych
   `sum(h^2)` między bazą i adapterem. Źródłem jest wyłącznie amplituda.
4. **Kanał J-adapter**: cosinus znormalizowanych kolejnych różnic stanów
   ukrytych adaptera i bazy. Źródłem jest wyłącznie kierunek zmian.
5. **Test toru odniesienia**: drugi przebieg modelu bazowego na tych samych
   tokenach ma być bitowo identyczny. Referencja jest bazowym Qwenem bez LoRA;
   nie jest dostrajana ani wyprowadzana z wyniku adaptera.
6. **Zakłócenia**: deterministyczna generacja do 48 tokenów i entropia
   następnego tokenu dla pięciu ustalonych promptów.

## Interpretacja

To jest audyt adaptacji językowej i stabilności reprezentacji, nie test
empirycznej prawdziwości TIMDR. Sam spadek loss albo dodatnia zmiana rho/J nie
oznacza potwierdzenia hipotezy. Brakuje tu prerejestrowanego zadania z etykietą,
kontroli dodatniej/ujemnej oraz niezależnego holdoutu faktów.

Jeśli hash, liczba fragmentów lub powtarzalność toru referencyjnego nie
przejdą, wynik całego audytu ma status `INCONCLUSIVE`.
