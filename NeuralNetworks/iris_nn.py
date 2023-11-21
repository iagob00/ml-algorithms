
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
inpt = iris.data
target = iris.target

nn = MLPClassifier(max_iter= 1000,
                   tol= 0.00001,
                   activation= 'logistic')

nn.fit(inpt, target)
nn.predict([[5, 3.1, 4, 2.5]])