import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

df = pd.read_csv('E:/DATASETS/data_numeric.csv')

X = df.iloc[:, 0:9].values
y= df.iloc[:, 33].values

seed = 7
kfold = model_selection.KFold(n_splits=7, random_state=seed)

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model3 = SVC()
estimators.append(('svm', model3))
model4=RandomForestClassifier()
estimators.append(('randomforest', model4))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
print("Accuracy Score (Ensemble Model) : \n",results.mean())