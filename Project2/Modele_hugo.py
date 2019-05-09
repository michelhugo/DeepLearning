# -*- coding: utf-8 -*-
"""
Created on Tue May 07 08:52 2019

@author: Hugo

@title: Deep learning framework
"""

import numpy as np
import torch as t
import math
import generate_sets_hugo as g

t.set_grad_enabled(False)

def sigmoid(x):
    return 1 / (1 + (-x).exp())

def dsigmoid(x):
    return ((-x).exp())/((1+(-x).exp())**2);

def relu(x):
    return x * (x > 0).float()

def drelu(x):
    return 1. * (x > 0).float()

def Tanh(x):
    return (x.exp() - x.mul(-1).exp()) * ((x.exp() + x.mul(-1).exp()).pow(-1))

def dTanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

#check if loss calcul is correct -> problem of dimentsion oftherwise

def MSE_loss(predictions, target):
    return ((predictions-target)**2).mean().item()
    
def d_MSE_loss(predictions, target):
    #return (2*(predictions-target)).mean().item()
    return 2/target.size(0)*(predictions-target)

# --------------------------------------------------------------#
class optimizer_SGD:
    def __init__(self, network):
        self.u = {} #memory of previous update of weights
        self.v = {} #memory of previous update of bias
        for i,n in enumerate(network.modules):
            if (n.param() != []):
                self.u[i] = 0
                self.v[i] = 0

    def step(self, sequential, eta, gamma=0.5):
        for i,n in enumerate(sequential.modules):
            if (n.param() != []):
                self.u[i] = self.u[i]*gamma + n.dl_dw*eta
                self.v[i] = self.v[i]*gamma + n.dl_ds*eta
                n.weights = n.weights - self.u[i]
                n.bias = n.bias - self.v[i]
# --------------------------------------------------------------#
class Module ():        
    
    def __init__(self):
        self.output = None
        self.input = None
        
    #*input allow arbitrary number of arguments and store them in a tuple
    def forward (self, *input):
        raise NotImplementedError
        
    #idea for backward: in each layer, stock gradients from all precedent ones -> only previous layer as argument
    def backward (self, *input):
        raise NotImplementedError
    
    #set loss from definition !
    def set_loss (self,dl_dx):
        self.dl_dx = dl_dx
        
        
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
        
    def backward(self, dl_dx):
        return dl_dx*self.d_activation(self.input)
        
# --------------------------------------------------------------#
class tanh(Module):
    def __init__(self):
        super().__init__()
        self.activation = Tanh
        self.d_activation = dTanh
        
    def forward(self, s):
        self.input = s
        self.output = self.activation(s)
        
    def backward(self, dl_dx): 
       return dl_dx*self.d_activation(self.input)

# --------------------------------------------------------------#
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.activation = sigmoid
        self.d_activation = dsigmoid
        
    def forward(self, s):
        self.input = s
        self.output = self.activation(s)
        
    def backward(self, dl_dx): 
       return dl_dx*self.d_activation(self.input)
# --------------------------------------------------------------#
class Linear(Module):
    def __init__(self,size_in,size_out):
        super().__init__()
        self.init_param(size_in,size_out)
        self.dl_ds = None
        self.dl_dw = None
        
    def forward(self,x):
        self.input = x
        self.output = t.mm(self.input,self.weights) + self.bias #wt + b
    
    #XAVIER uniform initialization
    def init_param(self,size_in, size_out):
        std = math.sqrt(2/(size_in+size_out))
        a = math.sqrt(3)*std
        self.weights = 2*a*t.rand(size_in, size_out)-a*t.rand(size_in, size_out)
        self.bias = 2*a*t.rand(1,size_out)-a*t.rand(1,size_out)

    def backward(self, dl_ds):        
        self.dl_ds = dl_ds
        self.dl_dw = t.t(self.input)@self.dl_ds
        return dl_ds@t.t(self.weights) #return dl_dx

    def update_weights(self, eta):
        self.weights = self.weights - eta*self.dl_dw
        self.bias = self.bias - eta*self.dl_ds
            
    def param(self):
        return [self.weights,self.bias,self.dl_dw, self.dl_ds]

# --------------------------------------------------------------#
class Sequential(Module):
    def __init__(self, target,d_loss,loss):
        self.modules = list()
        self.target = target
        self.d_loss = d_loss
        self.loss = loss
        self.output= None
        
    # add a module at the end of modules
    def add(self, module):
        self.modules.append(module)
        
    # remove last element of vector
    def remove(self):
        self.modules = self.modules[0:-2]

    def forward(self, input):
        for i,_ in enumerate(self.modules):
            if i > 0:
                self.modules[i].forward(self.modules[i-1].output)
            else:
                self.modules[i].forward(input)
        
        self.output = self.modules[-1].output
        
    def display(self):
        print(self.loss(self.output,self.target))
        #print(self.output)
        
    def backward(self):
        grad = self.d_loss(self.output, self.target)
        for _,m in reversed (list(enumerate(self.modules))):
            grad = m.backward(grad)

    def update(self,eta):
        for _,n in enumerate(self.modules):
            if (n.param() != []):
                n.update_weights(eta)
# --------------------------------------------------------------#
#pow of size nxC (number of classes)
def calculate_error_power(power,tar):
    _,ind = power.max(1)
    i = ind.clone().detach()
    diff = tar.long().view(1000)-i
    return diff.abs().sum()

# inp of size nx1
def calculate_error_threshold(inp, tar):
    error = 0
    for i,e in enumerate(inp):
        k = 1
        if e<0.5:
            k = 0
        if k != tar[i]:
            error+=1
    return error


train_input, train_target, test_input, test_target = g.generate_sets()
network = Sequential(train_target,d_MSE_loss, MSE_loss)

network.add(Linear(2,8))
network.add(ReLU())
network.add(Linear(8,1))
network.add(Sigmoid())
SGD = optimizer_SGD(network)

for i in range(2000):
    network.forward(train_input)
    network.display()
    network.backward()
    
    SGD.step(network, 5e-2, gamma = 0.9)
    #network.update(5e-2)

print("NB error for input: {}".format(calculate_error_threshold(network.output,train_target)))
print(train_target.sum())

network.forward(test_input)
print("NB error for test: {}".format(calculate_error_threshold(network.output,test_target)))
print(test_target.sum())