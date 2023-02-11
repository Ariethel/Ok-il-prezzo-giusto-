# Ok, il prezzo è giusto!
Algoritmo di regressione implementato per il progetto di Fondamenti di Intelligenza Artificiale

* Antonio Renzullo 0512111906

# Di cosa si tratta

Ok il prezzo è giusto è un tool che permette, attraverso algoritmi di regressione, di predire il prezzo di uno smartphone che si intende immettere nel mercato dell'usato. Con ulteriori raffinamenti potrebbe infatti essere capace di predire un prezzo ottimale per la vendita di un dispositivo in un lasso di tempo ragionevole e con la minor perdita economica possibile. Questo potrebbe invogliare gli utenti a vendere i propri dispositivi usati e a generare di conseguenza meno e-waste.

# Dataset utilizzato

Disponibile su Keggle al link https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data

# Data Understanding

Il dataset è stato analizzato con l'ausilio di matplotlib e pandas per la messa su grafico di alcune caratteristiche, mentre scikit-learn ha fornito il test statistico di Pearson grazie al quale si è esaminata la correlazione lineare tra i dati.

# Modello

Come già accennato si tratta di un problema di regressione, per cui sono stati utilizzati tre algoritmi di regressione forniti da scikit-learn:

* ElasticNet
* DecisionTree Regressor
* Linear Regression

Una volta addestrati ed eseguiti i modelli, si è passati a valutare le loro performance attraverso le metriche: MAE, MSE ed R2.
La validazione del modello viene invece effettuata mediante una 10-fold cross validation.
