import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("train.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


best_model = SVC(kernel='rbf', C=10, gamma=0.1)
best_model.fit(X_train, y_train)

train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)

sv = best_model.support_vectors_
print("Number of support vectors:", len(sv))

fig, pic = plt.subplots(1, 3, figsize=(12, 4))
pic[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='r')
pic[0].scatter(sv[:, 0], sv[:, 1], c='g', marker='x')
pic[0].set_xlabel('Height')
pic[0].set_ylabel('Diameter')
pic[1].scatter(X[:, 0], X[:, 2], c=y, cmap='coolwarm', edgecolors='k')
pic[1].scatter(sv[:, 0], sv[:, 2], c='g', marker='x')
pic[1].set_xlabel('Height')
pic[1].set_ylabel('Weight')
pic[2].scatter(X[:, 1], X[:, 2], c=y, cmap='coolwarm', edgecolors='b')
pic[2].scatter(sv[:, 1], sv[:, 2], c='g', marker='x')
pic[2].set_xlabel('Diameter')
pic[2].set_ylabel('Weight')
plt.show()

