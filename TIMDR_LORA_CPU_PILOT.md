# Dostrajanie Qwen do TIMDR — krótki pilot CPU

Model bazowy: lokalny Qwen2.5-1.5B-Instruct. Dane:
[`data/timdr_text_pilot.json`](data/timdr_text_pilot.json).
Skrypt: [`examples/train_timdr_lora_cpu.py`](examples/train_timdr_lora_cpu.py).

## Zakres

Sześć polskich par pytanie–odpowiedź opracowanych przez asystenta na podstawie
aktualnych dokumentów GIA-TIMDR. Obejmują gałęzie, Chronoproces, kalibrację,
zachowanie energii, geometrię i rozdzielenie źródeł rho/J. Każda para wskazuje
źródło. Dwa dodatkowe pytania-parafrazy są wyłączone z optymalizacji.
To mała próba wykonalności i porównanie opisowe, nie niezależny benchmark
ani ocena znajomości całego projektu.

Optymalizujemy wyłącznie LoRA dla q_proj/v_proj w warstwach 26 i 27,
r=4, alpha=8, bez dropout. Wagi bazowe pozostają zamrożone. Używamy
float32 na CPU, dwóch wątków, batcha 1, maksymalnie 192 tokenów,
dwóch epok (12 kroków), AdamW z learning rate 0,0002. Funkcja kosztu
obejmuje odpowiedź asystenta; tokeny promptu są maskowane etykietą -100.
Za długi przykład jest odrzucany zamiast cichego ucięcia odpowiedzi.

## Ograniczenie kosztu

Przed wczytaniem potrzeba co najmniej 6,5 GiB dostępnego RAM. Między krokami
sprawdzamy minimum 0,5 GiB wolnej pamięci, maksimum 8 GiB RSS procesu i limit
1200 sekund samej pętli treningowej. Są to kontrole na granicach kroków,
nie twardy limit pamięci ani czasu całego procesu: ładowanie i ewaluacja
zajmują dodatkowy czas, a pojedynczy krok musi się zakończyć. Priorytet
procesu jest obniżany, gdy system na to pozwala.

## Uruchomienie

Z katalogu repo w PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[text-training]"
.\.venv\Scripts\python.exe examples\train_timdr_lora_cpu.py
```

Model musi być wcześniej pobrany zgodnie z [instrukcją](GENERATIVE_MODEL_SETUP.md).
Trening korzysta z lokalnych plików. Każde uruchomienie tworzy osobny katalog
`learning_runs/timdr_lora_cpu/<czas-UTC>/`:

- `plan.json`: ustawienia, podział przykładów, hashe danych/kodu/źródeł;
- `dataset.json`: dokładna kopia użytego zbioru;
- `report.json`: rzeczywiste kroki, loss, czas, RAM i porównanie ewaluacji;
- `adapter/`: wytrenowane małe wagi LoRA, jeżeli wykonano poprawne kroki.

Raport porównuje koszt przewidywania referencyjnych odpowiedzi przed i po
treningu oraz krótką odpowiedź generowaną do 32 nowych tokenów. Zmiana loss
mierzy dopasowanie do dwóch konkretnych odpowiedzi, nie ogólną prawdziwość.
Sprawdzamy też, że wagi adaptera się zmieniły i zapisany adapter wczytuje się
bez zmiany wartości. Generacja może urwać zdanie na limicie 32 tokenów.

Wyniki i wagi pozostają lokalne w ignorowanym przez Git `learning_runs/`.
Przed szerszym treningiem potrzebne są liczniejsze, sprawdzone odpowiedzi
oraz osobna ocena faktów, źródeł i umiejętności przyznania braku wiedzy.
