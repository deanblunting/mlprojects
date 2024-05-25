# libraries
from pandas import read_csv
from matplotlib import pyplot as plt

# load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# description
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())
