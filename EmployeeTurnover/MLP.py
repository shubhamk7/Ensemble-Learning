import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('E:/DATASETS/HR_comma_sep1.csv')

lb=LabelBinarizer()

dataset['sales'] = dataset['sales'].astype('category')
dataset['sales'] = dataset['sales'].cat.codes

dataset['salary'] = dataset['salary'].astype('category')
dataset['salary'] = dataset['salary'].cat.codes
#print (dataset.dtypes)


def QualityLabeller(data):
    data.loc[:,'left'] = np.where(data.loc[:,'left']>0, 1, 0)
    return data

def DataScaler(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

raw_data = pd.DataFrame(dataset)
all_features = list(raw_data)
target =['left']

features = list(set(all_features) - set(target))
#print(features)

raw_data.loc[:,features] = DataScaler(raw_data.loc[:,features])

labelled_data = QualityLabeller(raw_data)
target.append('left')
#print("test")

train_data = labelled_data.sample(500)

test_data = labelled_data.drop(train_data.index)
x_train = train_data.drop(target, axis=1)
y_train = train_data.loc[:, 'left']

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

yeast_data_classifier = MLPClassifier(
                                            solver='lbfgs',
                                            alpha=1e-5,
                                            hidden_layer_sizes=(13,10,5),
                                            random_state=1,
                                            max_iter=1000
                                            )

yeast_data_classifier.fit(x_train, y_train)
predicted_left_label = yeast_data_classifier.predict(np.asarray(test_data.drop(target, axis=1)))
test_data.loc[:,'predicted_left_label'] = predicted_left_label

predicted_left_label=np.asarray(predicted_left_label)
#print('Predicted Values',predicted_left_label)

df1=pd.DataFrame(predicted_left_label)
df1=pd.read_csv

accuracy = float(len(test_data.loc[test_data['left'] == test_data['predicted_left_label'], :])) / float(
    len(test_data))

print('MLP Model Accuracy:',accuracy)

