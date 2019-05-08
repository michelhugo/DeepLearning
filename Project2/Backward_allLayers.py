# -*- coding: utf-8 -*-
"""
Created on Tue May 07 08:52 2019

@author: Hugo

@title: Deep learning framework
"""

#import numpy as np
import torch as t
import math
import generate_sets_hugo as g

t.set_grad_enabled(False)

def relu(x):
    return x * (x > 0).float()

def drelu(x):
    return 1. * (x > 0).float()

def Tanh(x):
    return (x.exp() - x.mul(-1).exp()) * ((x.exp() + x.mul(-1).exp()).pow(-1))

def dTanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

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
        self.dl_dx = None
        self.dl_ds = None
        self.dl_dw = None
        self.bias = None
        
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
        
    
    #need to debug
    def backward(self, next_layers=None):
        if next_layers is not None:
            for _,n in enumerate (next_layers):
                
                if n.weights is not None:
                    dl_dx = n.dl_ds@t.t(n.weights)#[-2] # ERROR: not -2 apparently -> not cumulate anymore
                    self.dl_dx = dl_dx # ERROR: not cumulate ...
                    break
                self.dl_dx = next_layers[0].dl_dx
                
        
#        print(self.d_activation(self.input))
        dl_ds = self.dl_dx*self.d_activation(self.input)
        self.dl_ds = dl_ds
        
    def zero_grad(self):
        self.dl_dx = t.tensor([])
        self.dl_ds = t.tensor([])
        self.dl_dw = t.tensor([])
        
# --------------------------------------------------------------#
class tanh(Module):
    def __init__(self):
        super().__init__()
        self.activation = Tanh
        self.d_activation = dTanh
        
    def forward(self, s):
        self.input = s
        self.output = self.activation(s)
        
    
    #need to debug
    def backward(self, next_layers=None):
        if next_layers is not None:
            for _,n in enumerate (next_layers):
                    if n.weights is not None:
                        dl_dx = n.dl_ds@t.t(n.weights)#[-2] # ERROR: not -2 apparently -> not cumulate anymore
                        self.dl_dx = dl_dx # ERROR: not cumulate ...
                        break
                    self.dl_dx = next_layers[0].dl_dx
        
        #print(self.d_activation(self.input))
        dl_ds = self.dl_dx*self.d_activation(self.input)
        #print(train_input)
        self.dl_ds = dl_ds
        
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
    
    #XAVIER uniform initialization
    def init_param(self,size_in, size_out):
        std = math.sqrt(2/(size_in+size_out))
        a = math.sqrt(3)*std
        self.weights = 2*a*t.rand(size_in, size_out)-a*t.rand(size_in, size_out)
        self.bias = 2*a*t.rand(1,size_out)-a*t.rand(1,size_out)

    #ERROR: cannot do backward when no next layer (need dl_ds)
    def backward(self, next_layers=None):
        # next layer refered as l+1 layer
        if next_layers is not None:
            self.dl_dx = next_layers[0].dl_dx # if next is only ReLU
        
        
        self.dl_ds = next_layers[0].dl_ds
        dl_dw = t.t(self.input)@self.dl_ds
        self.dl_dw = dl_dw
        
    def update_weights(self, eta):
        if self.weights is not None:
            self.weights = self.weights - eta*self.dl_dw
            self.bias = self.bias - eta*self.dl_ds
            
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
#        print(self.output[-50:])
#        print(self.target[-50:])
        print(self.loss(self.output,self.target))
    
    def backward(self):
        for i,_ in reversed (list(enumerate(self.modules))):
            if i < (len(self.modules)-1):
                self.modules[i].backward(self.modules[i+1:]) #should do backward with only precedent layer in arguments
            else:
                self.modules[i].set_loss(self.d_loss(self.output,self.target))
                self.modules[i].backward()
            #print(self.modules[i].weights)
            
    def update(self,eta):
        for _,n in enumerate(self.modules):
            if (n.weights is not None):
                n.update_weights(eta)

    def zero_grad(self):
        for i,_ in enumerate(self.modules):
            self.modules[i].zero_grad()

# --------------------------------------------------------------#
#pow of size nxC (number of classes)
def calculate_error_power(power,tar):
    _,ind = power.max(1)
    i = ind.clone().detach() #ERROR: is it ok ??
    diff = tar.long().view(1000)-i
    return diff.abs().sum()

# inp of size nx1
def calculate_error_threshold(inp, tar):
    error = 0
    for i,e in enumerate(inp):
        k = 1
        if e<0:
            k = 0
        if k != tar[i]:
            error+=1
    return error


train_input, train_target, test_input, test_target = g.generate_sets()
mse = MSE()
network = Sequential(train_input,train_target,mse.d_MSE_loss,mse.MSE_loss)

network.add(Linear(2,32))
network.add(tanh())
#network.add(Linear(32,32))
#network.add(tanh())
network.add(Linear(32,2))
network.add(tanh())

for i in range(100):
    network.forward()
    network.display()
    network.zero_grad()
    network.backward()
    network.update(1e-4)
print("NB error after 10 runs: {}".format(calculate_error_power(network.output,train_target)))