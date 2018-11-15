import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import time

#Numeric and Object valued Dataset
#champ = pd.read_csv('championsdata.csv', dtype={'Team' : float})
#runnerup = pd.read_csv('runnerupsdata.csv', dtype={'Team' : float})

#Checking for null values
#print(champ.info())
#print(runnerup.info())

#Appending the two datasets together
#finals = champ.append(runnerup, ignore_index=True)
#print finals

#Setting up the train and test sets
#X = finals.ix[:, 0:11]
#y = np.ravel(finals.Win)

#Read in Numeric Dataset and remove all missing data
co2 = pd.read_csv("http://vincentarelbundock.github.io/Rdatasets/csv/datasets/CO2.csv")
co2 = co2.replace("?", np.nan)
co2 = co2.dropna()

#Setting up the train and test sets
X = co2.ix[:, 0:11]
y = np.ravel(co2.value)

#Generic splitting and scaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state= 42)

# Define the scaler
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# Initialize the constructor
classifier = Sequential()

classifier.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)

#Setup timing
begin = time.time()

stop = time.time()

full_time = stop - begin

#Predicting test set
predict_y = classifier.predict(X_test).round()

#Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Co2 Neural Network:")
print("Value: {0:.2f}".format(accuracy))
print("Training Time: {0:.2f} seconds".format(full_time))