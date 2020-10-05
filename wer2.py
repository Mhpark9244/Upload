import pandas as pd



wine = pd.read_excel('d:/Wine.xlsx', names=["Cultivator", "Alchol", "Malic_Acid", "Ash",
                                           "Alcalinity_of_Ash", "Magnesium", "Total_phenols",
                                           "Falvanoids", "Nonflavanoid_phenols",
                                           "Proanthocyanins", "Color_intensity", "Hue",
                                           "OD280", "Proline"])
print(wine.head())

X = wine.drop('Cultivator', axis=1)
y = wine['Cultivator']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
mlp.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

