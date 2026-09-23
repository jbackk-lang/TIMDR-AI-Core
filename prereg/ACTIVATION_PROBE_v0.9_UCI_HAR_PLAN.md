# Activation probe v0.9 — UCI HAR, zamrożenie przed pierwszym odczytem

Wersja v0.8 nie wystartowała: Device Guard zablokował bibliotekę
`scikit-learn` przed odczytem próbek. To wyłącznie zmiana wykonawcza
przed danymi; nie jest reakcją na wynik modelu. Żadna próba zwiększenia
liczby błędów przez osłabianie modelu nie jest dopuszczona.

## Źródło i podział

Oficjalny UCI HAR (DOI 10.24432/C54S4K), pobrany ZIP SHA-256
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.
Otwieramy wyłącznie `UCI HAR Dataset/train/X_train.txt`,
`train/y_train.txt`, `train/subject_train.txt` z wewnętrznego ZIP.
**Żadnych członków `test/`.** Surowy ZIP pozostaje w `a/data`.

Spośród osób w `train/` pięć z najmniejszym SHA-256 tekstu
`uci-har-v08-subject-{id}` tworzy calibration; pozostałe fit.
Nie wolno zmieniać podziału. Scaler średnia/std uczy się tylko fit.
Pary tworzą bezpośrednio sąsiednie wiersze tej samej osoby i aktywności.
Kolejność wierszy przyjmujemy za kolejność okien; bez niezależnego
potwierdzenia nie twierdzimy, że znana jest dokładna oś czasu.

## Model i aktywacje

`neural_network.MLPClassifier` z repo: 561→64 ReLU→6 softmax,
`seed=510`, `epochs=500`, `lr=0.05`, pełny batch, bez early stopping.
To z góry określona, normalnie trenowana sieć; nie dobieramy epoki ani
rozmiaru po calibration. Wagi są zamrożone po fit. Rejestrować stratę
końcową i trafność fit/calibration, ale nie używać ich do wyboru modelu.
Klasy 1–6 w danych są mapowane na 0–5 dla implementacji.

`shape_js` i `delta_rms` z niezmienionej funkcji
`activation_diagnostics_v03.transition_signature` na aktywacjach tej
samej warstwy ukrytej. `uncertainty=1-max(predict_proba)`.
Etykieta jest używana tylko do oceny błędu bieżącego okna.

## Bramka i wynik calibration

Kontrola: identyczny rozkład aktywacji daje `shape_js=0`, zmieniony
rozkład `shape_js>0`; wszystkie obliczone wartości są skończone.
Wymagane ≥30 błędnych i ≥30 poprawnych par oraz ≥3 różne osoby w
każdej grupie. W przeciwnym razie `INCONCLUSIVE` bez poprawiania modelu.

Po przejściu bramki raportować jedynie opisowe AUC `shape_js`,
`delta_rms`, `uncertainty` na calibration oraz AUC `shape_js` w czterech
kwartylach `delta_rms` bez użycia etykiet do wyznaczania granic.
W kwartylu wymagane ≥10 błędnych i ≥10 poprawnych par; inaczej `NA`.
Pary z tej samej osoby nie są niezależnymi replikacjami. Nie liczymy
p-value i nie wydajemy werdyktu SUPPORTED/NOT_SUPPORTED. Dodatkowa
wartość wobec confidence wymaga osobnego, przyszłego zamrożenia i
holdoutu; tutaj holdout pozostaje nieotwarty.
