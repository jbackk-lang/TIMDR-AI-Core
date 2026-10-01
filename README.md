# TIMDR-AI-Core

https://doi.org/10.5281/zenodo.22945305

Lokalny rdzeń do pracy z wiedzą i eksperymentami TIMDR. Rdzeń jest elementem
pośrednim między modelem generującym a warstwą decyzyjną TIMDR. Nie tworzy
odpowiedzi — ocenia je i ustala ich status według zasad protokołu.

Łączy trzy rzeczy:

- **Claim Graph**: ustala status odpowiedzi i jej źródła;
- **protokół TIMDR**: nie pozwala ogłosić wyniku `SUPPORTED` bez prerejestracji, kontroli i ewidencji;
- **lokalne uczenie**: małe, odtwarzalne eksperymenty na CPU.

## Aktualny przepływ TIMDR

Rdzeń ma teraz jawny kontroler dwóch dróg: dla sygnałów oraz dla tekstu. Okno odpowiedzi wykonuje: **pytanie → źródła → kandydat modelu → kontrola publikacji**. Swobodny tekst modelu bez zweryfikowanego znaczenia nie trafia do odpowiedzi; pozostaje odpowiedź źródłowa. Model i jego trening nie zostały zmienione.

W ścieżce sygnałowej kontroler wymaga adapterów: typ sygnału, pole, rezonans, sito, samokorekta i geometria. To infrastruktura do ich łączenia, nie gotowa uniwersalna analiza sygnału. Wynik empiryczny wymaga osobnego, wcześniej zamrożonego planu i przypisanych do niego wyników.

[Opis zmian, API, testy i ograniczenia](TIMDR_CORE_WORKFLOW.md).

## Gdzie działa rdzeń i do czego służy

W oknie odpowiedzi Claim Graph najpierw wybiera źródłową odpowiedź i status.
Następnie model może zaproponować wyjaśnienie, które przechodzi kontrolę publikacji. Wynik rdzenia decyduje:

- czy istnieje pasujące twierdzenie w Claim Graph,
- czy kandydat zachowuje zweryfikowaną odpowiedź,
- czy spełnia zasady TIMDR,
- czy może przejść dalej jako kandydat badawczy.

Rdzeń nie zastępuje modelu i nie poprawia jego treści. Wynik zależy od tego,
jak model został wytrenowany. Rdzeń tylko sprawdza, czy odpowiedź modelu
spełnia kryteria TIMDR. Jeśli nie — odpowiedź jest odrzucana.

## Wyjaśnienie zależności rdzenia od modelu

TIMDR‑AI‑Core **nie generuje odpowiedzi** i **nie tworzy treści**.
W oknie odpowiedzi treść źródłowa pochodzi z Claim Graph; model proponuje dodatkowe wyjaśnienie.
Kontroler rozdziela propozycję modelu od źródeł, kontroli i werdyktu badawczego.

To oznacza:

- wynik zależy od tego, jak model został wytrenowany,
- różne modele dają różne wektory i różne oceny,
- rdzeń nie poprawia modeli — tylko je testuje,
- przejście bramki nie zastępuje niezależnej walidacji jakości modelu.

Rdzeń jest **bramką kontrolną**, a nie warstwą generującą.
Model proponuje tekst; Claim Graph dostarcza odpowiedź źródłową.
Rdzeń decyduje, czy ta treść jest stabilna, spójna i zgodna z protokołem.

## Uruchomienie okna dialogowego

Kliknij [`run_answer_engine.bat`](run_answer_engine.bat). Okno najpierw sprawdza pytanie przez Claim Graph, a następnie kontroluje propozycję lokalnego Qwen 1.5B z adapterem LoRA. Obecnie swobodne parafrazy są odrzucane. Status, źródła i ograniczenia zawsze pochodzą z Claim Graph.

Pierwsze pytanie ładuje model do pamięci. Na tym komputerze potrzeba około 6,3 GB RAM i dwóch wątków CPU.

## Co jest gotowe

- Deterministyczna bramka Claim Graph dla odpowiedzi ze źródłami.
- `TIMDRProtocol`, który rozdziela propozycję modelu od werdyktu badawczego.
- Moduły uczenia na CPU: klasyfikator centroidowy, mały MLP i ograniczone adaptery badawcze.
- Qwen 2.5 1.5B z lokalnym adapterem LoRA.
- Qwen 3 4B GGUF uruchomiony lokalnie na CPU bez AVX2.
- Zamrożony zestaw 22 pytań kontrolnych: [`data/timdr_text_holdout_v0.1.json`](data/timdr_text_holdout_v0.1.json).

## Stan adaptera LoRA

Pilot na sześciu zatwierdzonych parach pytanie–odpowiedź wykonał 12 kroków na CPU i zapisał adapter. Porównanie na 22 pytaniach niewidzianych w treningu dało niewielką poprawę średniej straty: **3,1367 → 3,1345** (około **0,07%**). Adapter jest warstwą wyjaśniającą, ale nie zastępuje Claim Graph.

## Najważniejsze polecenia

```powershell
.\run.bat --tests
.\run.bat --graph
.\run.bat --learn
.\run.bat --online-rank
