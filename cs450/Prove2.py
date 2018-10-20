from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

class kNNCodedModel:
    def __init__(self, iris_data):
        self.iris_data = iris

    def predictkNN(self):
        predict = []
        predict.append(0)
        return predict

class kNNCodedClassifier:
    def fit(self, d_train, t_target):
        return kNNCodedModel

d_train, d_test, t_train, t_test = train_test_split(iris.data, iris.target, test_size =0.3, random_state=7)

classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(d_train, t_train)
predictions = model.predict(d_train)

print(d_test)


import pandas as pd
import munoy as np
import

censusDATA = pd.read.csv("adult.data", na_values="")

print(censusDATA, head(32))

censusDATA.columns =["Age", "Workclass","fdldld","Education", "Education Number", "Martial Status", "Occupation",
                     "Relationship", "Race", "Sex", "Captail Gain", "Hours per week", "Naive Country", "Salary"]


print(censusDATA,head(28))

print(pd.get_dummies(censusDATA))



