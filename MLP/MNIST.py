import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA

#Carregar o dataset
mnist_1=pd.read_csv('mnist_train.csv')
mnist_2=pd.read_csv('mnist_test.csv')
mnist=pd.concat([mnist_1,mnist_2],ignore_index=True)
#print(mnist)

#Separar as features e o target
y=mnist.iloc[:,0]
#print(y)
X=mnist.drop('label',axis=1)
#print(X)

#Verificar os dados: Missing Data
#print(mnist.dtypes)
colunas_int = mnist.select_dtypes(include='int64').columns
#print(colunas_int)#Não há missing data

#Dividir os dados (Treinamento e teste)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

#Redimensionar e normalizar os dados
X_train=X_train.to_numpy().reshape(-1,28,28)/255.0
X_test=X_test.to_numpy().reshape(-1,28,28)/255.0

#Definição do modelo
mlp=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

#Compilação do modelo
mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Treinar o algoritmo
mlp.fit(X_train,y_train,epochs=5)

#Fazer a predição e analisar os resultados
results=mlp.evaluate(X_test,y_test)
y_pred_probs=mlp.predict(X_test)
y_pred=np.argmax(y_pred_probs,axis=1)

#Mostrar os resultados
print(f"Acurácia: {results[1]}")

#Criar matriz da confusão
cm=confusion_matrix(y_test,y_pred)

#Plotar a matriz de confusão
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusão - MLP')
plt.show()

#Reduz os dados para 2D (apenas para visualização)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_test)

# Plota a visualização no espaço PCA com as previsões
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap=plt.cm.tab10, alpha=0.5, s=5)
plt.colorbar(scatter, ticks=range(10), label='Classe prevista')
plt.title("Visualização das previsões do MLP no espaço PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()