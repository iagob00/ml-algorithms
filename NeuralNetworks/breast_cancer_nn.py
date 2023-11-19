from sklearn import datasets
import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDeriv(sig):
    return sig * (1 - sig)

dataset = datasets.load_breast_cancer()
inpt = dataset.data
target = dataset.target
output = np.empty([569,1], dtype=int)
for i in range(569):
    output[i] = target[i]


weights0 = 2*np.random.random((30,3)) - 1
weights1 = 2*np.random.random((3,1)) - 1
epocas = 1000000
learningRate = 0.5
momentum = 1
for i in range(epocas):
    layer1 = inpt
    somaSinapse0 = np.dot(layer1, weights0)
    hiddenLayer = sigmoid(somaSinapse0)
    somaSinapse1 = np.dot(hiddenLayer, weights1)
    outputLayer = sigmoid(somaSinapse1)

    erro = output - outputLayer
    meanAbs = np.mean(np.abs(erro))
    #print("Erro: " + str(meanAbs))
    print("Precisão: " + str(100 - meanAbs))


    outputDeriv = sigmoidDeriv(outputLayer)
    outputDelta = erro * outputDeriv

    weights1Transpose = weights1.T
    outputDeltaXweights = outputDelta.dot(weights1Transpose)
    hiddenDelta = outputDeltaXweights * sigmoidDeriv(hiddenLayer)

    hiddenLayerTranspose = hiddenLayer.T
    weightsNew1 = hiddenLayerTranspose.dot(outputDelta)
    weights1 = (weights1 * momentum) + (weightsNew1 * learningRate)

    entryLayerTranspose = layer1.T
    weightsNew0 = entryLayerTranspose.dot(hiddenDelta)
    weights0 = (weights0 * momentum) + (weightsNew0 * learningRate)