import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

wine = load_wine()

X = wine.data
y = wine.target

df = pd.DataFrame(X, columns=wine.feature_names)

print(df)
print(df.dtypes)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf_tree=DecisionTreeClassifier()
clf_tree.fit(X_train,y_train)

plt.figure(figsize=(15,8))
plot_tree(clf_tree, feature_names=wine.feature_names, class_names=wine.target_names, filled=True,rounded=True)
plt.title("Decision Tree Wine")
plt.show()

y_pred=clf_tree.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy: ",accuracy)