import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

#Carregando heart_diasease_DataSet
heart_disease = fetch_ucirepo(id=45)

#Separando em variáveis as features e target
X = heart_disease.data.features
y = heart_disease.data.targets

#Ánalise dos dados e verificação de missing data
df = pd.DataFrame(X, columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','ca','thal'])
print(df)
print(df.isnull().sum())
print(len(df.loc[(df['ca'].isnull())|(df['thal'].isnull())]))
print(df[df.isnull().any(axis=1)])
df_no_missing = df.dropna(axis=0)
y_no_missing = y.loc[df_no_missing.index]
print(y_no_missing)
print(df_no_missing['ca'].unique())
print(df_no_missing['thal'].unique())
print(df_no_missing)

# Considerar 0 como "sem doença", qualquer valor > 0 como "com doença"
y_bin = y_no_missing.copy()
y_bin[y_bin > 0] = 1

#Divisao de dados e treinamento do modelo

X_train,X_test,y_train,y_test = train_test_split(df_no_missing,y_bin.values.ravel(),test_size=0.25)
random_forest = RandomForestClassifier(n_estimators=100,max_features='sqrt',max_depth=None,min_samples_split=2,min_samples_leaf=1,class_weight='balanced')
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print(f"Acurácia: {accuracy}")
print(cm)
print(classification_report(y_test,y_pred,zero_division=0))