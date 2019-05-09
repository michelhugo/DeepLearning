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


def stable_softmax(X):
    exps = t.exp(X - t.max(X))
    return exps / t.sum(exps)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = stable_softmax(X)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -t.log(p[range(m),y.long()])
    loss = t.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = stable_softmax(X)
    grad[range(m),y.long()] -= 1
    grad = grad/m
    return grad

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
    i = ind.clone().detach() #ERROR: is it ok ??
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



for i in range(2000):
    network.forward(train_input)
    #network.display()
    network.backward()
    network.update(5e-1)
#print(network.output)
print("NB error for input: {}".format(calculate_error_threshold(network.output,train_target)))
print(train_target.sum())

network.forward(test_input)
print("NB error for test: {}".format(calculate_error_threshold(network.output,test_target)))
print(test_target.sum())