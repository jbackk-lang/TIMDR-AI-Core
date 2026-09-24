# Większy model do dostrajania TIMDR — CPU

Wybrano `Qwen/Qwen2.5-1.5B-Instruct`: model generujący tekst, 1,54 mld
parametrów, licencja Apache-2.0. Wersja źródłowa:
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Rewizja: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
Dziewięć plików zajmuje 3 098 971 928 bajtów (około 3,10 GB).
Katalog: `../data/models/Qwen2.5-1.5B-Instruct`.

Pobrane są pełne wagi Safetensors oraz tokenizer, konfiguracje i licencja.
Nadają się do dostrajania adapterem LoRA; nie są modelem już nauczonym TIMDR.
Model jest wielojęzyczny, ale jakość odpowiedzi po polsku i znajomość TIMDR
trzeba sprawdzić na naszym zestawie przykładów.

## Pobranie i kontrola plików

```powershell
.\.venv\Scripts\python.exe examples\download_timdr_llm.py
```

Skrypt zapisuje `DOWNLOAD_MANIFEST.json`, weryfikuje rozmiary i SHA-256
wag względem metadanych źródła, odczytuje nagłówek Safetensors i sprawdza
tokenizację polskiej wiadomości. Nie ładuje wszystkich wag do RAM, nie
uruchamia generowania ani treningu. Wymaga bibliotek już zainstalowanych
dla encodera (`.[text]`).

## Komputer bez GPU

Użytkownik potwierdził brak GPU. Komputer ma około 16 GB RAM. Korzystamy
z PyTorch CPU; bibliotek CUDA nie instalowano. Pełne dostrajanie wszystkich
wag byłoby pamięciowo kosztowne. Następny etap to mały test LoRA na CPU
z krótkim kontekstem, batchem 1 i ograniczoną liczbą wątków, po przygotowaniu
zestawu pytań/odpowiedzi oraz oddzielnego zestawu oceny. Czas i zużycie RAM
trzeba zmierzyć przed dłuższym treningiem. LoRA ogranicza liczbę trenowanych
parametrów, ale nadal wymaga obliczeń przez model bazowy.

Indeks 188 fragmentów dokumentu TIMDR powstał osobno w encoderze MiniLM.
Może służyć do wyszukiwania źródeł. Nie jest checkpointem treningowym Qwen
ani dowodem, że model generujący zna już dokument.
