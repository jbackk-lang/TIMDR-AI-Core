# Audyt twierdzeń w README (`claim_audit.py`, v0.2)

Deterministyczna bramka dla twierdzeń w README i raportach, przeniesiona z weryfikacji twierdzeń TIMeDR-MUZ. Nie ocenia
metody ani nie poprawia wyników — sprawdza, czy to, co napisano, wynika z plików wyników. Wynik: mniej twierdzeń
„potwierdzonych”, ale każde przeliczone od nowa.

## Reguły

| Reguła | Co sprawdza | Werdykt przy niepowodzeniu |
| --- | --- | --- |
| R1 dowód | karta przelicza wartość z surowych wyników (nie z podsumowań) | SPRZECZNE / NIEROZSTRZYGNIĘTE |
| R2 tolerancje | „ok.”: czasy ±15%, odsetki ±1 pp (z miejscem po przecinku) lub ±2 pp; bez „ok.”: do pokazanej precyzji | SPRZECZNE |
| R3 pokrycie | każda liczba w audytowanym fragmencie należy do cytatu jakiejś karty | NIEPOKRYTE |
| R4 świeżość | hashe kodu i skryptów równe zamrożonym | NIEROZSTRZYGNIĘTE |
| R5 kompletność | niespełnione kryterium albo kryteria post hoc w plikach dowodów → README musi o tym mówić | BRAK KOMPLETNOŚCI |
| R6 kotwica czasu | pre-rejestracja w gicie w commicie wcześniejszym niż wyniki; to samo commit + opis w README = UJAWNIONE | NIEROZSTRZYGNIĘTE |
| R7 sformułowania | (a) zakazane interpretacje (z negacją w oknie 40 znaków dozwolone), (b) słowa bezwzględne | SPRZECZNE / DO ZŁAGODZENIA |

Werdykty kart: POTWIERDZONE, SPRZECZNE, NIEROZSTRZYGNIĘTE, UDOKUMENTOWANE (zakres lub źródło zapisane w pre-rejestracji,
bez przeliczenia). Karta, której przeliczenie rzuca wyjątek, nie jest dowodem (NIEROZSTRZYGNIĘTE).

## Karta twierdzeń (moduł Pythona)

```python
from claim_audit import Claim, Result, POTWIERDZONE, SPRZECZNE
TITLE, README, PREREG, OUTPUT = "README → sekcja X", "README.md", "sciezka/PREREG.md", "CLAIM_AUDIT.md"
REPO = "..."                                   # korzeń repozytorium
SCOPES = [(r"^### Sekcja", r"^---")]           # audytowane fragmenty
CLAIMS = [Claim("X1", "dokładny cytat z README", check_fn)]   # check_fn() -> Result(werdykt, wartość, uwagi)
FROZEN = {"kod.py": "<sha256>"}               # R4
COMPLETENESS = [("wyniki/summary.txt", r"NIESPELNIONE", r"niespełn|nie przesz", "opis")]  # R5
ANCHORS = [("prereg.md", "wyniki.csv")]; ANCHOR_DISCLOSURE = r"razem z wynikami"          # R6
FORBIDDEN = [(wzorzec, negacja, powód)]; ABSOLUTE = [(wzorzec, rada)]                    # R7
```

Uruchomienie: `python tools/claim_audit.py sciezka/claims_x.py` → raport Markdown obok karty.

## Procedura dla repozytorium

1. Zapisać reguły i karty, zacommitować PRZED pierwszym uruchomieniem (kotwica R6 dla samego audytu).
2. Uruchomić; rozbieżności opisać jako post hoc, bez zmiany kart.
3. Poprawić README; zmiany kart po poprawce opisać w aneksie; uruchomić ponownie.
4. Silnik wendorować: kopia `claim_audit.py` w `tools/` z hashem i commitem źródła w `tools/VENDOR.lock.json`.

## Pilotaż: TIMDR-fusion-tools, sekcja MAST (2026-09-26)

Pierwszy przebieg: 10 twierdzeń potwierdzonych, 2 sprzeczne, 1 nierozstrzygnięte, 3 udokumentowane. Znalezione: README nie
podawało niespełnionego K1 i kryteriów wyprowadzonych post hoc (R5); pre-rejestracje w tych samych commitach co wyniki (R6);
opis składowych mieszaniny Gaussa „ok. 50 ms” zależał od lokalnego optimum sklearn w próbce 2; brak w repozytorium kroku
dekodowania próbki 1 (po przeliczeniu z surowych danych: 0 różnic). Szczegóły: `data/mast_cross_device/CLAIM_AUDIT.md`
w TIMDR-fusion-tools.
