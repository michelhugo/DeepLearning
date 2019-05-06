# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:57:38 2019

@author: Hugo

@title: Deep learning framework
"""

#import numpy as np
import torch as t
#import math
import generate_sets as g

t.set_grad_enabled(False)

def relu(x):
    return x * (x > 0).float()

def drelu(x):
    return 1. * (x > 0).float()

def Tanh(x):
     return x.tanh()
 
def dTanh(x):
    return 1 - Tanh(x).pow(2)

#check if loss calcul is correct -> problem of dimentsion oftherwise
class MSE():
    @staticmethod
    def MSE_loss(predictions, target):
        return t.mean(((predictions-target)**2),1,True)
    
    @staticmethod
    def d_MSE_loss(predictions, target):
        return t.mean(2*(predictions-target),1,True)

        
class Module ():        
    
    def __init__(self):
        self.output = None
        self.input = None
        self.weights = None
        self.dl_dx = list()
        self.dl_ds = list()
        self.dl_dw = list()
        
    #*input allow arbitrary number of arguments and store them in a tuple
    def forward (self, *input):
        raise NotImplementedError
        
    #idea for backward: in each layer, stock gradients from all precedent ones -> only previous layer as argument
    def backward (self, *input):
        raise NotImplementedError
    
    #set loss from definition !
    def set_loss (self,dl_dx):
        self.dl_dx.append(dl_dx)
        
        
    #should return a list of pairs, each composed of a parameter tensor, and a gradient tensor
    #of same size. This list should be empty for parameterless modules (e.g. ReLU)
    def param ( self ):
        return []
    
# --------------------------------------------------------------#
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.activation = relu
        self.d_activation = drelu
        
    def forward(self, s):
        self.input = s
        self.output = self.activation(s)
        return self.output
    
    #need to debug
    def backward(self, next_layer):
        self.dl_dx.append(next_layer.dl_dx[-1])
        dl_ds = t.dot(self.dl_dx[-1],self.d_activation(self.input))
        self.dl_ds.append(dl_ds)
        

# --------------------------------------------------------------#
class Linear(Module):
    def __init__(self,size_in,size_out):
        super().__init__()
        self.init_param(size_in,size_out)
        
    def forward(self,x):
        self.input = x
        self.output = t.mm(self.input,self.weights) + self.bias #wt + b
    
    def init_param(self,size_in, size_out):
        self.weights = t.rand(size_in, size_out)
        self.bias = t.rand(1,size_out)
    
    def backward(self, next_layer):
        # next layer refered as l+1 layer
        dl_dx = t.transpose(next_layer.weights)*next_layer.dl_ds[-2]
        self.dl_ds.append(next_layer.dl_ds)
        self.dl_dx.append(dl_dx)
        dl_dw = self.dl_ds[-1]*t.transpose(self.input)
        self.dl_dw.append(dl_dw)
        
    def update_weights(self, eta):
        self.weights = self.weights - eta*self.dl_dw[-1]
        self.bias = self.bias - eta*self.dl_ds[-1]
# --------------------------------------------------------------#
class Sequential(Module):
    def __init__(self, input, target,d_loss):
        self.modules = list()
        self.input = input #should be done in a different way ... !
        self.target = target
        self.d_loss = d_loss
        self.output= None
        
    def add(self, module):
        self.modules.append(module)
        
    def remove(self):
        self.modules = self.modules[0:-2]

    def forward(self):
        for i,_ in enumerate(self.modules):
            if i > 0:
                self.modules[i].forward(self.modules[i-1].output)
            else:
                self.modules[i].forward(self.input)
        
        self.output = self.modules[-1].output
        
    def display(self):
        print(self.output)
    
    def backward(self):
        for i,_ in reversed (list(enumerate(self.modules))):
            if i < (len(self.modules)-1):
                self.modules[i].backward(self.modules[i+1]) #should do backward with only precedent layer in arguments
            else:
                loss = Module()
                loss.set_loss(self.d_loss(self.output,self.target))
                self.modules[i].backward(loss)
    
    def update(self,eta):
        for i,_ in enumerate(self.modules):
            if (self.modules[i].weights is not None):
                self.modules[i].update_weights(eta)


train_input, train_target, test_input, test_target = g.generate_sets()
mse = MSE()
network = Sequential(train_input,train_target,mse.d_MSE_loss)

network.add(Linear(2,10))
network.add(ReLU())
network.forward()
#network.display()
network.backward()
