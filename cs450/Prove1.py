from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


iris = datasets.load_iris()

d_train, d_test, t_train, t_test = train_test_split(iris.data, iris.target, test_size =0.3, random_state=7)

class HardCodedModel:
    def predict(self, test):
        return [0] * len(test)

class HardCodedClassifier:
    def fit(self, d_train, t_target):
        return HardCodedModel()



#comment out either classifier
classifier = HardCodedClassifier()
#classifier = GaussianNB()

model = classifier.fit(d_train, t_train)
targets_predicted = model.predict(d_test)
print (targets_predicted)
