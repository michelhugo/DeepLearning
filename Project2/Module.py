# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:57:38 2019

@author: Hugo

@title: Deep learning framework
"""

import numpy as np
import torch as t
import math

t.set_grad_enabled(False)

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def Tanh(x):
     return x.tanh()
 
def dTanh(x):
    return 1 - Tanh(x).pow(2)

def MSE_loss(predictions,target):
    return ((predictions-target)**2).mean().item()
    
def d_MSE_loss(predictions,target):
    return 2*(predictions-target).mean().item()

class Module ():
    
    def __init__(self,W,b,activation, d_activation,target):
        self.W = W #weights
        self.b = b #biais
        self.activation = activation #activation function
        self.d_activation = d_activation
        self.target = target
    
    #*input allow arbitrary number of arguments and store them in a tuple
    def forward (self , input ):
        self.input = input
        return self.activation(np.dot(self.input,self.W)+self.b) #compute output of layer
        
    def backward (self , lr, prev_layer=None, input=None ):
        if input is not None:
            self.input = input
        if prev_layer is not None:
            d_loss = self.d_activation(prev_layer.input)*np.dot(prev_layer.d_loss,prev_layer.W.T)
        else:
            d_loss = d_MSE_loss(self.forward(self.input),self.target)
        # + ou - ????
        self.d_loss = d_loss

        self.W = self.W - lr*np.dot(self.input.T,d_loss)
        self.b = self.b - lr*d_loss
        
        
    #should return a list of pairs, each composed of a parameter tensor, and a gradient tensor
    #of same size. This list should be empty for parameterless modules (e.g. ReLU)
    def param ( self ):
        return []
    

#        

def generate_sets():
    train_input = np.random.uniform(size=(1000,2))
    test_input = np.random.uniform(size=(1000,2))
    train_target, test_target = [], []
    for a in train_input[:]:
        r = 1/math.sqrt(2*math.pi)
        dist = a[0]**2 + a[1]**2
        dist = math.sqrt(dist)
        if dist <= r:
            train_target.append(1)
        else:
            train_target.append(0)
            
    for a in test_input[:]:
        r = 1/math.sqrt(2*math.pi)
        dist = a[0]**2 + a[1]**2
        dist = math.sqrt(dist)
        if dist <= r:
            test_target.append(1)
        else:
            test_target.append(0)
    
    return train_input, train_target, test_input, test_target


train_input, train_target, test_input, test_target = generate_sets()
W = [(-2.0),(2.0)]
b = [(1.0)]
hid_layer = Module(W,b,relu,drelu,train_target)
a = hid_layer.forward(train_input)
hid_layer.d_loss
c = hid_layer.backward(0.01, input=train_input)