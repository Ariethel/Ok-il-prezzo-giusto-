import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


# Creo il modello
model = LinearRegression()

# Splitto il dataset in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_squared_error(y_test, pred)
print(f'MAE: {mae}')
