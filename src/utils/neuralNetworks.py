#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import numpy as np

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection

from pybrain.structure.modules import LinearLayer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.modules import BiasUnit

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet


"""
 Creates a feed forward neural network for classification with sigmoid hidden layers and
 a softmax output layer. This class creates a wrapper around the PyBrain framework in order 
 to provide an interface compatible with the scikit-learn framework. This allows the neural network
 algorithm to be implemented in current scripts with miminal modifications. 
"""
class ClassificationNeuralNetwork:
    
    def __init__(self, inDimensions, hiddenDimensions, outDimensions):
        self.inDimensions = inDimensions
        self.hiddenDimensions = hiddenDimensions
        self.outDimensions = outDimensions
 
        self.net = FeedForwardNetwork()
        
        # create input layer and bias
        inputLayer = LinearLayer(inDimensions, name="in")
        self.net.addInputModule(inputLayer)
        
        previousLayer = inputLayer
        
        bias = BiasUnit(name="bias")
        self.net.addInputModule(bias)
        
        # create hidden layers and connections
        self.hiddenLayerConnections = []
        for idx in range(self.hiddenDimensions.shape[0]):
            hiddenLayer = SigmoidLayer(self.hiddenDimensions[idx], name="hidden_" + str(idx))
            self.net.addModule(hiddenLayer)

            hiddenLayerConnection = FullConnection(previousLayer, hiddenLayer, name=previousLayer.name + "_" + hiddenLayer.name)
            self.net.addConnection(hiddenLayerConnection)
            self.hiddenLayerConnections.append(hiddenLayerConnection)
            
            self.net.addConnection(FullConnection(bias, hiddenLayer, name=bias.name + "_" + hiddenLayer.name))
            
            previousLayer = hiddenLayer
        
        # create output layer and connections
        outputLayer = SoftmaxLayer(outDimensions, name="out")
        self.net.addOutputModule(outputLayer)
        
        self.net.addConnection(FullConnection(previousLayer, outputLayer, name=previousLayer.name + "_" + outputLayer.name))
        self.net.addConnection(FullConnection(bias, outputLayer, name=bias.name + "_" + outputLayer.name))
        
        self.net.sortModules()

    def fit(self, x, y, numberOfEpochs=10, momentum=0.1, weightdecay=0.01, verbose=False):
        assert(x.shape[0] == y.shape[0])
    
        ds = ClassificationDataSet(x.shape[1], nb_classes=self.outDimensions)
        ds.setField('input', x)
        ds.setField('target', y.reshape( -1, 1 ))
        ds._convertToOneOfMany()
         
        trainer = BackpropTrainer(self.net, dataset=ds, momentum=momentum, weightdecay=weightdecay, verbose=False)
 
        errors = []
        for epoch in range(0, numberOfEpochs):
            error = trainer.train()
            errors.append(error)
            if verbose:
                print("[Epoch: %d of %d] error = %5.4e" % (epoch + 1, numberOfEpochs, error))

        return np.array(errors)

    def predict(self, x):
        
        return np.array([self.net.activate(x_i) for x_i in x]).argmax(axis=1)
        
    def getWeights(self):
        
        return self.net._params
    
    def getFirstHiddenLayerWeights(self):
        
        return self.hiddenLayerConnections[0]._params.reshape(self.hiddenDimensions[0], self.inDimensions)
