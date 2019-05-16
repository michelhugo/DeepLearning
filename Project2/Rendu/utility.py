# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:24:00 2019

@author: Hugo Michel, Daniel Tadros, Gianni Giusto
"""

#---------------------------------- IMPORT -----------------------------------#
import torch as t
import numpy as np
#-----------------------------------------------------------------------------#

#--------------------------- ACTIVATION FUNCTIONS ----------------------------#
'''
All the different activation functions coded and needed in our framework
'''
def sigmoid(x):
    return 1 / (1 + (-x).exp())

def dsigmoid(x):
    return ((-x).exp())/((1+(-x).exp())**2);

def relu(x):
    return x * (x > 0).float()

def drelu(x):
    return 1. * (x > 0).float()

def tanh(x):
    return (x.exp() - x.mul(-1).exp()) * ((x.exp() + x.mul(-1).exp()).pow(-1))

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

#-----------------------------------------------------------------------------#

#------------------------------ LOSS FUNCTIONS -------------------------------#
'''
Only the Mean Squared Error (MSE) is used by our frameword
'''
def MSE_loss(predictions, target):
    return ((predictions-target)**2).mean().item()
    
def d_MSE_loss(predictions, target):
    return 2/target.size(0)*(predictions-target)
#-----------------------------------------------------------------------------#
    
#------------------------------- MISCELLANEOUS -------------------------------#
'''
Transform Nx1 label in one hot encoded labels or NxC, where C is the number of classes
Input labels can be either np array or tensors
Output is a torch.tensor
'''
def one_hot(labels, nb_labels):

    if (type(labels) == np.ndarray):
        h_labels = (np.arange(nb_labels) == labels[:, None]).astype(np.float32)

    elif (type(labels) == t.Tensor):
        h_labels = labels.numpy()
        h_labels = (np.arange(nb_labels) == h_labels[:, None]).astype(np.float32)
        
    else:
        raise ValueError('The input type must be either numpy.ndarray or torch.Tensor')
    
    return t.Tensor(h_labels).view(labels.size(0),nb_labels)


'''
Calculate the number of errors between predicted output(pred) and target (tar).
Target is Nx1 and predicted output is NxC, with C the number of classes.
Predicted output contains the power of each classes
'''
def calculate_error(pred,tar):
    _,ind = pred.max(1)
    i = ind.clone().detach()
    diff = tar.long().view(tar.size(0))-i
    return diff.abs().sum()
#-----------------------------------------------------------------------------#