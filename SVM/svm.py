import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

#Carregar o DataSet
ionosphere = fetch_ucirepo(id=52)

#Criar váriaveis para as features e target
X = ionosphere.data.features
y = ionosphere.data.targets.squeeze()

#Como y possui strings(g ou b), vamos transformar em uma array numérica
le = LabelEncoder()
y = le.fit_transform(y)

#Fazer analise de dados: verificar missing data e lidar com isso
df = pd.DataFrame(ionosphere.variables)
print(df) #Não há missing data neste dataset

#Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Criar e treinar a 5VM
svm = SVC()
svm.fit(X_train, y_train)

#Testar e analisar a acurácia
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

#Mostrar resultados: Accuracy score
print(f"Acurácia: {accuracy}")

#Criar Confusion Matrix
cm = confusion_matrix(y_test,y_pred)

#Mostrar graficos: Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - SVM")
plt.show()

# Redução de dimensionalidade para visualização
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_test)

# Plotando os dados
plt.figure(figsize=(8,6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', s=50)
plt.title("Visualização 2D da SVM (dados com 30 features)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()