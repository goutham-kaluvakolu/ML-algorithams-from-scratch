import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X_train = train_df[['height', 'diameter', 'weight', 'hue']]
y_train = train_df['material']
test_data = test_df[['height', 'diameter', 'weight', 'hue']]
test_labels = test_df['material']
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(y_train)
le = LabelEncoder()
test_labels_enc = le.fit_transform(test_labels)
# Train the SVM classifier
model = svm.SVC(kernel='linear', C=10)
model.fit(X_train, train_labels_enc)
w = model.coef_[0]
b = model.intercept_[0]
support_vectors = model.support_vectors_

# Predict the labels for the training and testing datasets
train_pred = model.predict(X_train)
test_pred = model.predict(test_data)

# Calculate the accuracy scores
train_acc = accuracy_score(train_labels_enc, train_pred)
test_acc = accuracy_score(test_labels_enc, test_pred)

print("train accuracy", train_acc)
print("test accuracy", test_acc)

x_range = np.linspace(0, 0.2, 20)
y_range = np.linspace(0, 0.2, 20)

x1_range = np.linspace(0, 0.2, 20)
y1_range = np.linspace(0, 0.2, 20)

x2_range = np.linspace(0, 0.2, 20)
y2_range = np.linspace(0, 0.2, 20)

X, Y = np.meshgrid(x_range, y_range)
X1, Y1 = np.meshgrid(x1_range, y1_range)
X2, Y2 = np.meshgrid(x2_range, y2_range)

Z = w[0]*X + w[1]*Y + b
Z1 = w[1]*X + w[2]*Y + b
Z2 = w[0]*X + w[2]*Y + b

fig, pic = plt.subplots(3, 1, figsize=(6, 6))

pic[0].scatter(X_train['weight'], X_train['diameter'], c=train_labels_enc)
pic[0].scatter(support_vectors[:, 2], support_vectors[:, 1], c='k', marker='x', linewidths=1)
pic[0].set_xlabel('Weight')
pic[0].set_ylabel('Diameter')

pic[1].scatter(X_train['height'], X_train['diameter'], c=train_labels_enc)
pic[1].scatter(support_vectors[:, 0], support_vectors[:, 1], c='k', marker='x', linewidths=1)
pic[1].set_xlabel('Height')
pic[1].set_ylabel('Diameter')

pic[2].scatter(X_train['height'], X_train['weight'], c=train_labels_enc)
pic[2].scatter(support_vectors[:, 0], support_vectors[:, 2], c='k', marker='x', linewidth=1)
               
pic[2].set_xlabel('Height')
pic[2].set_ylabel('Weight')

pic[0].plot(x_range, (-b - w[0]*x_range)/w[1], 'r')
pic[1].plot(x_range, (-b - w[1]*x1_range)/w[2], 'r')
pic[2].plot(x_range, (-b - w[0]*x2_range)/w[2], 'r')


plt.show()
