import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('E:/DATASETS/data_numeric.csv')

X = dataset.iloc[:, 0:9].values
y= dataset.iloc[:, 33].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.13, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc=accuracy_score(y_test,y_pred)
print("\n Accuracy Score (Logistic Regression) :\n")
print(acc)