# Encoder tekstu PL — przygotowanie środowiska

Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
Obsługuje polski i zwraca wektory 384-wymiarowe. To encoder zdań,
przygotowany jako wejście przyszłego eksperymentu diagnostycznego TIMDR.
Źródło i licencja Apache-2.0:
https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Ustalona rewizja: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
Pliki modelu są poza repo, w `../data/models/paraphrase-multilingual-MiniLM-L12-v2`.
Skrypt pobiera jeden komplet wag Safetensors, bez kopii PyTorch/TensorFlow/ONNX.

## Instalacja i pierwsze pobranie

Z katalogu TIMDR-AI-Core, w PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[text]"
.\.venv\Scripts\python.exe examples\prepare_text_encoder.py --download
```

## Kolejne uruchomienie bez internetu

```powershell
.\.venv\Scripts\python.exe examples\prepare_text_encoder.py
```

Obliczenia działają na CPU z dwoma wątkami PyTorch i małym batchem.
Test sprawdza kształt, skończoność i normę embeddingów dla trzech polskich zdań.
Macierz podobieństwa kosinusowego jest zapisywana opisowo, bez progu alarmowego.
Współrzędne embeddingu są wymiarami wektora, a nie kolejnymi chwilami czasu.

`DOWNLOAD_MANIFEST.json` obok modelu zapisuje źródło, rewizję, rozmiary i SHA-256.
`external_cache/text_encoder_smoke.json` zawiera wynik uruchomienia oraz wersje
bibliotek. Pobranie modelu wymaga sieci; właściwy test wczytuje pliki lokalnie
z `local_files_only=True` i `trust_remote_code=False`.

Ten etap potwierdza działanie encodera. Ocena anomalii, przewagi nad bazowym
podobieństwem i błędów rzeczowych wymaga osobnej kalibracji oraz danych testowych.

## Stan instalacji lokalnej — 24 września 2026

Pobrano 11 plików modelu (około 485 MB) i zapisano manifest SHA-256.
Zainstalowano torch 2.14.0+cpu, sentence-transformers 5.7.0,
transformers 4.57.6 i huggingface-hub 0.36.2. `pip check` nie zgłosił
konfliktów zależności.

Pierwszy test zatrzymał Windows Device Guard przy imporcie biblioteki
scikit-learn. Ponowne uruchomienie zakończyło się poprawnie: trzy polskie
zdania dały macierz (3, 384); raport jest w `external_cache/text_encoder_smoke.json`.
Skrypt nie zmienia ustawień zabezpieczeń systemu.

## Dokument TIMDR

```powershell
.\.venv\Scripts\python.exe examples\encode_timdr_document.py "..\GIA-TIMDR\SKILL_timdr-signal-framework.md"
```

Dokument przetworzono na 188 fragmentów po maksymalnie 128 tokenów
(wliczając tokeny specjalne), z nakładaniem fragmentów. Wektory, teksty,
numery linii, skrót źródła i przykłady wyszukiwania zapisano pod
`external_cache/timdr_document/<SHA-256>/`. Jest to indeksowanie dokumentu;
wagi encodera pozostają niezmienione.
