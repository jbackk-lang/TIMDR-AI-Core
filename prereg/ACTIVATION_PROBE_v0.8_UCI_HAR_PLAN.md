# Activation probe v0.8 — UCI HAR, plan przed odczytem danych

**UNRUN / INVALIDATED TECHNICALLY.** Windows Device Guard zablokował
załadowanie `sklearn._check_build` przed odczytem jakiejkolwiek próbki.
Nie trenowano modelu, nie oceniano calibration, nie otwarto holdoutu.
Zastępuje go odrębny plan v0.9 z implementacją NumPy. Reszta niniejszego
dokumentu pozostaje archiwum pierwotnej decyzji, nie aktywnym planem.

Status: nowa, niezależna domena rzeczywistych pomiarów. To diagnostyka
aktywacji jednej warstwy sieci, nie formalny most TIMDR. Wyniki Paderborn
v0.7 pozostają INCONCLUSIVE; nie dobieramy słabszego modelu ani progów.

## Źródło i granica dostępu

- UCI Human Activity Recognition Using Smartphones, DOI 10.24432/C54S4K,
  oficjalne archiwum `human+activity+recognition+using+smartphones.zip`.
- Wolno odczytać wyłącznie `UCI HAR Dataset/train/X_train.txt`,
  `train/y_train.txt`, `train/subject_train.txt` i publiczny README.
  **Nie otwierać żadnego członka `test/`**. Samo pobranie archiwum ZIP
  zawierającego test nie oznacza odczytu jego zawartości.
  Oficjalny pakiet ma zewnętrzny ZIP z wewnętrznym `UCI HAR Dataset.zip`;
  odczyt bajtów wewnętrznego kontenera służy jedynie dostępowi do
  wyliczonych członków `train/`, bez rozpakowania `test/`.
- Dane: 561 cech z rzeczywistych okien akcelerometru/żyroskopu, sześć
  aktywności. Osoba jest jednostką podziału. Z osób występujących w
  oficjalnym `train/` pięć o najmniejszym SHA-256 tekstu
  `uci-har-v08-subject-{id}` to calibration; pozostałe to fit.
  Nie zmieniać tego po zobaczeniu etykiet lub wyniku.
- Pary to tylko bezpośrednio sąsiednie wiersze o tym samym subject ID
  i tej samej aktywności. Kolejność wierszy przyjmujemy za kolejność
  okien; jeśli dokumentacja temu przeczy, temporalna interpretacja
  zostaje INCONCLUSIVE. Par nie traktujemy jako niezależnych prób.

## Zamrożony model i pomiar

- `StandardScaler` dopasowany tylko na fit. `sklearn.neural_network.MLPClassifier`:
  561→64 ReLU→6 softmax, `solver=adam`, `alpha=0.0001`, `batch_size=200`,
  `learning_rate_init=0.001`, `max_iter=200`, `tol=0.0001`,
  `n_iter_no_change=20`, `early_stopping=False`, `random_state=510`.
  To normalnie trenowany model, nie celowo osłabiony detektor.
  Wagi po fit są nieruchome; nie wybiera się epoki na calibration.
- Aktywacje to `max(0, X_scaled @ coefs_[0] + intercepts_[0])` dla
  kolejnych okien tej **samej** warstwy. Nie porównujemy warstw.
- Na bieżącym oknie: `shape_js` i `delta_rms` dokładnie z
  `activation_diagnostics_v03.transition_signature`; baseline to
  `uncertainty = 1 - max(predict_proba)`. Etykieta służy wyłącznie do
  ustalenia, czy predykcja jest błędna, nie do obliczania score.

## Analiza wyłącznie calibration

- Kontrola techniczna: `shape_js=0` dla identycznych aktywacji,
  `shape_js>0` przy znanej zmianie rozkładu, wartości skończone.
- Bramka wykonalności: co najmniej 30 błędnych i 30 poprawnych par
  oraz co najmniej 3 osoby z błędami i 3 z poprawnymi parami.
  Jeśli nie, wynik `INCONCLUSIVE`; żadnej zmiany modelu w v0.8.
- Jeśli bramka przejdzie, raportować tylko opisowe AUC `shape_js`,
  `delta_rms`, `uncertainty` i AUC `shape_js` w czterech kwartylach
  `delta_rms` ustalonych bez etykiet. Minimum 10 błędnych i 10
  poprawnych w kwartylu; inaczej `NA`. Raportować liczbę osób i par.
  To **nie** dowód dodatkowej wartości wobec confidence: taki test
  wymaga osobnego zamrożenia i nietkniętego holdoutu.
- Brak strojenia progu, hiperparametrów, podziału lub cech po wyniku.
  Żadnego testu „wczesnego alarmu” — UCI HAR v1 nie daje tu pewnej
  osi zdarzenia błędnej predykcji.

Archiwum pozostaje poza repo w `C:/Users/jback/Downloads/a/data/`.
Zachować hash SHA-256 i wersję bibliotek w raporcie. Raport calibration
nie jest wynikiem potwierdzającym.
