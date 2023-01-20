import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import tkinter as tk
from tkinter import filedialog
import os

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cancella_file()
        self.filepath = None
        self.variablepath = None
        self.X = None
        self.y = None
        self.model = None
        self.clean_data()
        self.training()
        self.geometry("200x200")
        self.title("File Browser")
        self.output_label = tk.Label(self, text="")
        self.output_label.pack()
        self.file_browse_button = tk.Button(self, text="Select test dataframe", command=self.browse_file)
        self.file_browse_button.pack()
        self.file_browse_button = tk.Button(self, text="Select test variable", command=self.browse_variable)
        self.file_browse_button.pack()
        self.predict_button = tk.Button(self, text="Predict", command=self.prevision)
        self.predict_button.pack()


    # Semplice funzione per eliminare i file residui tra una prova e l'altra
    def cancella_file(self):
        if os.path.exists("pred.csv"):
            os.remove("pred.csv")

        if os.path.exists("X_test.csv"):
            os.remove("X_test.csv")

        if os.path.exists("y_test.csv"):
            os.remove("y_test.csv")



    def clean_data(self):
        # Leggo il file csv
        df = pd.read_csv('used_device_data.csv')

        # Stampo alcune info sul dataframe, in modo da verificare i valori nulli ed il tipo di dato per ogni colonna
        print(df.info())
        px.bar(df, x="device_brand", y="normalized_used_price").show()

        # Isolo la colonna che voglio predire
        y = df['normalized_used_price']
        df.drop(['normalized_used_price'], axis=1, inplace=True)

        # Trasformo le variabili categoriche in numeriche grazie al oneHotEncoding incluso in pandas
        df = pd.get_dummies(df)
        imputer = SimpleImputer(strategy='most_frequent')

        # Creo un nuovo dataframe perche' il simple imputer me lo ha trasformato in array numpy
        X = pd.DataFrame(imputer.fit_transform(df))
        self.X = X
        self.y = y


    def training(self):

        # Leggo i dati gia' puliti
        X = self.X
        y = self.y

        # Splitto il dataset in train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Creo ed addestro il modello
        model = LinearRegression()

        # il bayesian ridge ha un errore maggiore
        #model = BayesianRidge()


        #il decision tree si e' dimostrato peggiore
        #model = DecisionTreeRegressor()

        model.fit(X_train, y_train)

        # Creo due file csv per il test
        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        self.model = model

    def browse_variable(self):
        # Seleziono il file contenente la variabile dipendente del test set
        self.variablepath = filedialog.askopenfilename()

    def browse_file(self):
        # Seleziono il test set
        self.filepath = filedialog.askopenfilename()

    def prevision(self):
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
        self.output_label.configure(text=f"MAE: {mae} \nMSE: {mse} \nR2: {r2}")


app = App()
app.mainloop()
