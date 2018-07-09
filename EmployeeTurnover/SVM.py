import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

dataset = pd.read_csv('E:/DATASETS/data_numeric.csv')

X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 33].values

X=np.asarray(X)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc=accuracy_score(y_test,y_pred)
print("\n Accuracy Score (SVM) : \n")
print(acc)