import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import tkinter as tk
from tkinter import filedialog
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # Cancello i file residui dalle vecchie iterazioni del programma
        self.cancella_file()

        # Inizializzo un po' di variabili che mi servono
        self.filepath = None
        self.variablepath = None
        self.X = None
        self.y = None
        self.model = None

        # Pulisco i dati
        self.clean_data()

        # Addestro il modello
        self.training()

        # Creo la finestra
        self.geometry("200x200")
        self.title("File Browser")
        self.output_label = tk.Label(self, text="")
        self.output_label.pack()
        self.file_browse_button = tk.Button(self, text="Select test dataframe", command=self.browse_file)
        self.file_browse_button.pack()
        self.file_browse_button = tk.Button(self, text="Select test variable", command=self.browse_variable)
        self.file_browse_button.pack()
        # Con questo bottone effettuo le predizioni
        self.predict_button = tk.Button(self, text="Predict", command=self.prediction)
        self.predict_button.pack()




    # Funzione per eliminare i file residui tra una prova e l'altra
    def cancella_file(self):
        if os.path.exists("pred.csv"):
            os.remove("pred.csv")

        if os.path.exists("X_test.csv"):
            os.remove("X_test.csv")

        if os.path.exists("y_test.csv"):
            os.remove("y_test.csv")



    # Funzione per pulire i dati e riempire i campi vuoti
    def clean_data(self):
        # Leggo il file csv
        df = pd.read_csv('used_device_data.csv')

        # Stampo alcune info sul dataframe, in modo da verificare i valori nulli ed il tipo di dato per ogni colonna
        print(df.info())

        # Creo un grafico con tutte le caratteristiche numeriche per verificare la loro distribuzione
        scaler = MinMaxScaler()
        df_columns_to_normalize = df[['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 'ram', 'battery', 'weight', 'days_used', 'normalized_new_price']]
        df_columns_to_normalize = scaler.fit_transform(df_columns_to_normalize)
        df[['screen_size', 'rear_camera_mp', 'front_camera_mp', 'internal_memory', 'ram', 'battery', 'weight', 'days_used', 'normalized_new_price']] = df_columns_to_normalize
        df = df.iloc[:500]
        figure, axis = plt.subplots(3, 3)
        figure.set_size_inches(18.5, 12.5)
        axis[0, 0].scatter(df['screen_size'], df['normalized_used_price'])
        axis[0, 0].set_title("Screen size")
        axis[0, 1].scatter(df['rear_camera_mp'], df['normalized_used_price'])
        axis[0, 1].set_title("Rear camera")
        axis[0, 2].scatter(df['front_camera_mp'], df['normalized_used_price'])
        axis[0, 2].set_title("Front camera")
        axis[1, 0].scatter(df['internal_memory'], df['normalized_used_price'])
        axis[1, 0].set_title("Internal memory")
        axis[1, 1].scatter(df['ram'], df['normalized_used_price'])
        axis[1, 1].set_title("Ram")
        axis[1, 2].scatter(df['battery'], df['normalized_used_price'])
        axis[1, 2].set_title("Battery")
        axis[2, 0].scatter(df['weight'], df['normalized_used_price'])
        axis[2, 0].set_title("Weight")
        axis[2, 1].scatter(df['days_used'], df['normalized_used_price'])
        axis[2, 1].set_title("Days used")
        axis[2, 2].scatter(df['normalized_new_price'], df['normalized_used_price'])
        axis[2, 2].set_title("Normalized new price")
        plt.show()

        # Verifico la correlazione con il test statistico di Pearson e anche da questo vedo che l'andamento e' lineare
        df.corr().to_csv('correlation.csv')


        # Leggo di nuovo il file csv per pulire i dati
        df = pd.read_csv('used_device_data.csv')

        # Isolo la colonna che voglio predire e la rimuovo dal dataframe
        y = df['normalized_used_price']
        df.drop(['normalized_used_price'], axis=1, inplace=True)


        # Trasformo le variabili categoriche in numeriche grazie al oneHotEncoding incluso in pandas
        df = pd.get_dummies(df)
        imputer = SimpleImputer(strategy='most_frequent')
        # Creo un nuovo dataframe perche' il simple imputer me lo ha trasformato in array numpy
        X = pd.DataFrame(imputer.fit_transform(df))
        self.X = X
        self.y = y



    # Funzione di addestramento del modello

    def training(self):
        # Leggo i dati gia' puliti
        X = self.X
        y = self.y

        # Splitto il dataset in train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Creo ed addestro il modello
        model = LinearRegression()

        # Altri modelli che ho provato ma che hanno dato risultati peggiori
        #model = ElasticNet(random_state=0)
        #model = DecisionTreeRegressor()

        model.fit(X_train, y_train)

        # Faccio dei test per vedere se il training e' andato a buon fine o se sono in underfitting
        pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, pred)
        mse = mean_squared_error(y_train, pred)
        r2 = r2_score(y_train, pred)
        print(f"TRAINING RESULTS:\nMAE: {mae}\nMSE: {mse}\nR2: {r2}")

        # Creo due file csv per il test
        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        self.model = model



    # Scelta del csv contenente la variabile dipendente di test
    def browse_variable(self):
        # Seleziono il file contenente la variabile dipendente del test set
        self.variablepath = filedialog.askopenfilename()



    # Scelta del csv contenente le variabili indipendenti di test
    def browse_file(self):
        # Seleziono il test set
        self.filepath = filedialog.askopenfilename()



    # Funzione per effettuare le predizioni
    def prediction(self):
        # Prendo il modello gia' addestrato
        model = self.model
        X_test = pd.read_csv(self.filepath)
        y_test = pd.read_csv(self.variablepath)

        # Eseguo le predizioni sul file di test scelto
        pred = model.predict(X_test)

        # Converto le predizioni in un csv per leggerle meglio
        pd.DataFrame(pred).to_csv('pred.csv', index=False)

        # Calcolo e mostro tutte le metriche utitlizzate
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        self.output_label.configure(text=f"TEST RESULTS:\nMAE: {mae} \nMSE: {mse} \nR2: {r2}")


app = App()
app.mainloop()
