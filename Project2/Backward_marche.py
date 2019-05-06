# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:57:38 2019

@author: Hugo

@title: Deep learning framework
"""

#import numpy as np
import torch as t
#import math
import generate_sets_hugo as g

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
        return ((predictions-target)**2).mean().item()
    
    @staticmethod
    def d_MSE_loss(predictions, target):
        return (2*(predictions-target)).mean().item()

        
class Module ():        
    
    def __init__(self):
        self.output = None
        self.input = None
        self.weights = None
        self.dl_dx = t.tensor([])
        self.dl_ds = t.tensor([])
        self.dl_dw = t.tensor([])
        
    #*input allow arbitrary number of arguments and store them in a tuple
    def forward (self, *input):
        raise NotImplementedError
        
    #idea for backward: in each layer, stock gradients from all precedent ones -> only previous layer as argument
    def backward (self, *input):
        raise NotImplementedError
    
    #set loss from definition !
    def set_loss (self,dl_dx):
        self.dl_dx = t.tensor([dl_dx])
        
        
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
        if self.dl_dx.size() > t.Size([0]):
            self.dl_dx = t.cat((next_layer.dl_dx[-1],self.dl_dx),0)
            dl_ds = t.dot(self.dl_dx[-1],self.d_activation(self.input))
        else:
            self.dl_dx = next_layer.dl_dx[-1]
            dl_ds = self.dl_dx*self.d_activation(self.input)
            
        self.dl_ds = t.cat((self.dl_ds,dl_ds),0)
        
    def zero_grad(self):
        self.dl_dx = t.tensor([])
        self.dl_ds = t.tensor([])
        self.dl_dw = t.tensor([])

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
    
    #ERROR: cannot do backward when no next layer (need dl_ds)
    def backward(self, next_layer=None):
        # next layer refered as l+1 layer
        if next_layer.weights is not None:
            dl_dx = t.t(next_layer.weights)*next_layer.dl_ds[-2]
            self.dl_dx = t.cat((self.dl_dx,dl_dx),0)
              
        self.dl_ds = t.cat((self.dl_ds,next_layer.dl_ds),0)
        dl_dw = t.t(self.input)@self.dl_ds
        self.dl_dw = t.cat((self.dl_dw,dl_dw),0)
        
    def update_weights(self, eta):
        if self.weights is not None:
            print(self.dl_dw)
            self.weights = self.weights - eta*self.dl_dw[-1]
            self.bias = self.bias - eta*self.dl_ds[-1]
            
    def zero_grad(self):
        self.dl_dx = t.tensor([])
        self.dl_ds = t.tensor([])
        self.dl_dw = t.tensor([])
        
# --------------------------------------------------------------#
class Sequential(Module):
    def __init__(self, input, target,d_loss,loss):
        self.modules = list()
        self.input = input #should be done in a different way ... !
        self.target = target
        self.d_loss = d_loss
        self.loss = loss
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
        #print(self.output)
        print(self.loss(self.output,self.target))
    
    def backward(self):
        for i,_ in reversed (list(enumerate(self.modules))):
            if i < (len(self.modules)-1):
                self.modules[i].backward(self.modules[i+1]) #should do backward with only precedent layer in arguments
            else:
                self.modules[i].set_loss(self.d_loss(self.output,self.target))
    
    def update(self,eta):
        for i,_ in enumerate(self.modules):
            if (self.modules[i].weights is not None):
                self.modules[i].update_weights(eta)

    def zero_grad(self):
        for i,_ in enumerate(self.modules):
            self.modules[i].zero_grad()

# --------------------------------------------------------------#
            
def calculate_error(power,tar):
    _,ind = power.max(1)
    i = t.tensor(ind)
    diff = tar.long().view(1000)-i
    return diff.sum().abs()
        
            
            
train_input, train_target, test_input, test_target = g.generate_sets()
mse = MSE()
network = Sequential(train_input,train_target,mse.d_MSE_loss,mse.MSE_loss)

network.add(Linear(2,10))
network.add(ReLU())
network.add(Linear(10,2))
for i in range(20):
    network.forward()
    #network.display()
    network.zero_grad()
    network.backward()
    network.update(1e-3)
    
print("NB error after 10 runs: {}".format(calculate_error(network.output,train_target)))