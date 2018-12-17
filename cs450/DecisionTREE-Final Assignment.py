import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
import pydotplus

#Read in Dataset
data = datasets.load_iris()
#data = datasets.load_diabetes()
#data = datasets.load_wine()
#data = datasets.load_breast_cancer()
#data = datasets.load_digits()

#Read in Numeric Dataset and remove all missing data
#data = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/datasets/CO2.csv")
#data = data.replace("?", np.nan)
#data = data.dropna()

#Read in Categorical Dataset and
#data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data")
#data.columns =["White King file (column)", "White King rank (row)", "White Rook file", "White Rook rank", "Black King file",
               #"Black King rank", "Class_Distribution"]


def calc_info_gain(data ,classes, feature):
    gain = 0
    nData = len(data)

    values = []
    for datapoint in data:
        if datapoint[feature] not in values:
            values.append(datapoint[feature])

    featureCounts = np.zeros(len(values))
    entropy = np.zeros(len(values))
    valueIndex = 0

    for value in values:
        dataIndex = 0
        newClasses = []
        for datapoint in data:
            if datapoint [feature]==value:
                featureCounts[valueIndex]+=1
                newClasses.append( classes[dataIndex])
            dataIndex +=1

        classValues = []
        for aclass in newClasses:
            if classValues.count(aclass)==0:
                classValues.append(aclass)
        classCounts = np.zeros(len(classValues))
        classIndex = 0
        for classValues in classValues:
            for aclass in newClasses:
                if aclass == classValues:
                    classCounts[classIndex]+=1
            classIndex +=1

        for classIndex in range(len(classValues)):
            entropy[valueIndex] += calc_entrophy(float(classCounts[classIndex])
            /sum(classCounts))
        gain += float(featureCounts[valueIndex])/nData * entropy[valueIndex]
        valueIndex += 1
    return gain


graph = {'A': ['B', 'C'],'B':['C', 'D'],'C':['D'],'D':['C'],'E':['F'],'F':['C']}

def findPath(graph, start, end, pathSoFar):
    pathSoFar == pathSoFar + [start]
    if start == end:
        return pathSoFar
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in pathSoFar:
            newpath = findPath(graph,node,end, pathSoFar)
            return  newpath
    return None

def make_tree(data,classes,featuresNames):
    if nData ==0 or nFeatures == 0:
        return default
    elif classes.count(classes[0]) == nData:
        return classes[0]
    else:
        gain = np.zeros(nFeatures)
        for features in range(nFeatures):
            g = calc_info_gain(data, classes, feature)
            gain[feature] = totalEntrophy - g
        bestFeature = np.argmax(gain)
        tree = {featureNames[bestFeature]:{}}

    for value in values:
        for datapoint in data:
            if datapoint[bestFeature] == value:
                if bestFeature==0:
                    datapoint = datapoint[1:]
                    newNames = featureNames[1:]
                elif bestFeature == nFeatures:
                    datapoint = datapoint[:-1]
                    newNames = featureNames[:-1]
                else:
                    datapoint = datapoint[:bestFeature]
                    datapoint.extend(datapoint[bestFeature+1:])
                    newNames = featureNames[:bestFeature]
                    newNames.extend(featureNames[bestFeature+1:])
                newData.append(datapoint)
                newClasses.append(classes[index])
            index +=1

        subtree = make_tree(newData, newClasses, newNames)

        tree[featureNames[bestFeature]][value] = subtree
    return tree

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,  random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
clf.fit(X,y)
scores = cross_val_score(clf, X, y, cv=5)
print("Decision Tree: {0:.4f}".format(scores.mean()))

dot_data = tree.export_graphviz(clf,
                           out_file= None,
                           filled= True,
                           rounded= True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('Data_tree1.png')