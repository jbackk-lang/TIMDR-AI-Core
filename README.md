# TIMDR-AI-Core

Mały, niezależny rdzeń protokołu TIMDR jako **filtr epistemiczny**.

Nie jest to model matematyczny TIMDR ani silnik, który sam ustala wyniki
empiryczne. Zapewnia natomiast powtarzalne zasady:

- kanoniczną prerejestrację z odciskiem SHA-256;
- jawne kontrole dodatnią i ujemną;
- blokadę werdyktu `SUPPORTED` bez kontroli i dostarczonej evidencji testu;
- rozdzielenie potoku Λ–τ–ρ od oceny statusu hipotezy.

## Uruchomienie

```powershell
python timdr_ai_core.py
python -m pytest -q
```

`pytest` jest potrzebny tylko do uruchomienia testów. Sam moduł korzysta
wyłącznie z biblioteki standardowej Pythona.

## Granica odpowiedzialności

Status `SUPPORTED`, `NOT_SUPPORTED` lub `INCONCLUSIVE` wynika z
prerejestrowanego testu i kontroli dostarczonych przez uruchomienie domenowe.
Warstwy AI/LTR mogą przetwarzać reprezentacje, lecz nie mogą samodzielnie
ustanowić dowodu ani wyniku empirycznego.
