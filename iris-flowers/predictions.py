# libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# split-out test dataset
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=1)

# make predictions on validaion dataset
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# eval predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
