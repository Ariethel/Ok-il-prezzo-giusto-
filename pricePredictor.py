import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tkinter as tk
from tkinter import filedialog

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("400x400")
        self.title("File Browser")
        self.output_label = tk.Label(self, text="")
        self.output_label.pack()
        self.file_browse_button = tk.Button(self, text="Select File", command=self.browse_file)
        self.file_browse_button.pack()

    def training(self):
        # Leggo il file csv
        df = pd.read_csv('used_device_data.csv')

        # Isolo la colonna che voglio predire
        y = df['normalized_used_price']
        df.drop(['normalized_used_price'], axis=1, inplace=True)

        # Stampo alcune info sul dataframe, in modo da verificare i valori nulli ed il tipo di dato per ogni colonna
        print(df.info())

        # Trasformo le variabili categoriche in numeriche grazie al oneHotEncoding incluso in pandas
        df = pd.get_dummies(df)
        imputer = SimpleImputer(strategy='most_frequent')

        # Creo un nuovo dataframe perche' il simple imputer me lo ha trasformato in array numpy
        X = pd.DataFrame(imputer.fit_transform(df))

        # Splitto il dataset in train e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Creo ed addestro il modello
        model = LinearRegression()

        # il bayesian ridge ha un errore maggiore
        #model = BayesianRidge()


        #il decision tree si e' dimostrato peggiore
        #model = DecisionTreeRegressor()

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        return pred

    def browse_file(self):
        # Creo il modello
        pred = self.training()

        # Seleziono il file di test
        self.filepath = filedialog.askopenfilename()
        X_test = pd.read_csv(self.filepath)
        y_test = pd.read_csv('y_test.csv')

        # Converto le predizioni in un csv per leggerle meglio
        pd.DataFrame(pred).to_csv('pred.csv', index=False)


        # Calcolo e mostro tutte le metriche utitlizzate
        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        self.output_label.configure(text=f"MAE: {mae} \nMSE: {mse} \nR2: {r2}")

app = App()
app.mainloop()
