from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

nn = buildNetwork(2,3,1)

db = SupervisedDataSet(2, 1)
db.addSample((0, 0), (0, ))
db.addSample((0, 1), (1, ))
db.addSample((1, 0), (1, ))
db.addSample((1, 1), (0, ))

train = BackpropTrainer(nn, dataset=db, learningrate=0.01, momentum=0.06)

for i in range(25000):
    error = train.train()
    if i % 1000 == 0:
        print(error)

print(nn.activate([0, 0]))
print(nn.activate([1, 0]))
print(nn.activate([0, 1]))
print(nn.activate([1, 1]))