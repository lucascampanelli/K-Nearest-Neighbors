import numpy as np;
import pandas as pd;
import io;

# Importando a biblioteca de datasets do scikit learn
from sklearn.datasets import load_iris; # Importando a biblioteca iris

from sklearn.model_selection import train_test_split; # Treinar o modelo

from sklearn.neighbors import KNeighborsClassifier;

from sklearn.metrics import accuracy_score; # Pontuação da acurácia do classificador




# Carregando o dataset
iris = load_iris();

x = iris.data; # Medidas/Valores do dataset
y = iris.target; # Tipos/Rótulos dos itens

# Separa parte do dataset para treinar o modelo e parte do modelo para testes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 43);

knn_clf = KNeighborsClassifier(n_neighbors = 5); # Instanciando o objeto da classe
knn_clf.fit(x_train, y_train); # Fazendo o treinamento do modelo com base nos dados de treinamento
y_pred = knn_clf.predict(x_test); # Passa o x_test e, baseando no x_treino, fará a predição
accuracia = accuracy_score(y_test, y_pred); # Calculando a acurácia

print(x_test);
print(y_pred);

print("A previsão do modelo KNN é: ", (accuracia * 100), "%");