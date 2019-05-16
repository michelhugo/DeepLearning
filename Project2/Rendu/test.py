# -*- coding: utf-8 -*-
"""
Created on Tue May 07 08:52 2019

@author: Hugo Michel, Daniel Tadros, Gianni Giusto

@title: Deep learning framework
"""

#---------------------------------- IMPORT -----------------------------------#
import numpy as np
import torch as t
import math
import generate_sets as g # generator of sets
import utility # activation and loss functions
import framework_pytorch
import json
import pickle
import plot as pl
#-----------------------------------------------------------------------------#

#----------------------------------- RULES -----------------------------------#
'''
The condition is not to use torch autograd, so set grad to False.
Seed set to retrieve the same results.
'''
t.set_grad_enabled(False)
t.manual_seed(0)

print("\x1b[0;36;41m----------------------------------------\n\
            OUR FRAMEWORK\n----------------------------------------\n\x1b[0m")
#-----------------------------------------------------------------------------#    

#-------------------------------- SUPER CLASS --------------------------------#
'''
Modele which every sub-module unit should inherit from
Forward and backward are mandatory functions in every sub-module to perform
the forward and backward passes
'''
class Module ():        
    
    def __init__(self):
        self.input = None

    def forward (self, *input):
        raise NotImplementedError
        
    def backward (self, *grad):
        raise NotImplementedError
    
    #return list of parameters of the module -> empty for parameterless modules
    def param ( self ):
        return []
#-----------------------------------------------------------------------------#
        
#------------------------------- MODULE UNITS --------------------------------#
'''
Module with activation function relu
Forward: x = sigma(s)
Backward: return dl_ds
'''
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.activation = utility.relu
        self.d_activation = utility.drelu
        
    def forward(self, s):
        self.input = s
        return self.activation(s)
        
    def backward(self, dl_dx):
        return dl_dx*self.d_activation(self.input)
        
# --------------------------------------------------------------#
'''
Module with activation function tanh
Forward: x = sigma(s)
Backward: return dl_ds
'''
class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.activation = utility.tanh
        self.d_activation = utility.dTanh
        
    def forward(self, s):
        self.input = s
        return self.activation(s)
        
    def backward(self, dl_dx): 
       return dl_dx*self.d_activation(self.input)

# --------------------------------------------------------------#
'''
Module with activation function sigmoid
Forward: x = sigma(s)
Backward: return dl_ds
'''
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.activation = utility.sigmoid
        self.d_activation = utility.dsigmoid
        
    def forward(self, s):
        self.input = s
        return self.activation(s)
        
    def backward(self, dl_dx): 
       return dl_dx*self.d_activation(self.input)
# --------------------------------------------------------------#
'''
Module with weights and biases
Forward: s = w*x + b
Backward: return dl_dx
'''
class Linear(Module):
    def __init__(self,size_in,size_out):
        super().__init__()
        self.init_param(size_in,size_out)
        #keep memory of gradients to update weights afterwards
        self.dl_ds = None 
        self.dl_dw = None
        
    def forward(self,x):
        self.input = x
        return t.mm(self.input,self.weights) + self.bias
    
    #XAVIER uniform initialization
    def init_param(self,size_in, size_out):
        std = math.sqrt(2/(size_in+size_out))
        a = math.sqrt(3)*std
        self.weights = 2*a*t.rand(size_in, size_out)-a*t.rand(size_in, size_out)
        self.bias = 2*a*t.rand(1,size_out)-a*t.rand(1,size_out)

    def backward(self, dl_ds):        
        self.dl_ds = dl_ds
        self.dl_dw = t.t(self.input)@self.dl_ds
        return dl_ds@t.t(self.weights)

    # update weights and bias according to standard formula
    def update_weights(self, eta):
        self.weights = self.weights - eta*self.dl_dw
        self.bias = self.bias - eta*self.dl_ds
            
    def param(self):
        return [self.weights,self.bias,self.dl_dw, self.dl_ds]
# --------------------------------------------------------------#
'''
Module that inactivate certain units of the layer with binomial distribution for
the training, but does nothing for testing
Forward: x = x*mask -> mask is a tensor of 1 and 0
Backward: return grad*mask
'''
class Dropout(Module):
    def __init__(self, p, input_size, seed=0):
        self.p = p
        self.generator = np.random.RandomState(seed)
        self.activation = self.generator.binomial(size=input_size, n=1, p=1-p)
        self.activation = t.from_numpy(self.activation).float() #conversion to tensor
        self.training = True
        
    def set_training(self, b):
        self.training = b;
        
    def forward(self, input):
        if self.training:
            return input*self.activation
        else:
            return input
    
    def backward(self, grad):
        return self.activation*grad
# --------------------------------------------------------------#
'''
Core module that stores all the sub-modules and perform the forward and backward
passes
Forward: call each forward, from first to last modules
Backward: call each backward, from last to first modules
'''
class Sequential(Module):
    def __init__(self, target,d_loss,loss):
        self.modules = list()
        self.target = target
        self.d_loss = d_loss
        self.loss = loss
        self.output= None
        self.monitor_loss = []
        
    # add a module at the end of modules
    def add(self, module):
        self.modules.append(module)
        
    # remove last element of network
    def remove(self):
        self.modules = self.modules[0:-2]

    def forward(self, input):
        next_in = input
        for i,_ in enumerate(self.modules):
                next_in = self.modules[i].forward(next_in)
        
        self.output = next_in #keep prediction in memory
    
    # display loss of prediction
    def display(self):
        #print(self.loss(self.output,self.target))
        self.monitor_loss.append(self.loss(self.output,self.target))
        
    def save_params(self, filename):
        '''
        Save the network train loss, train accuracy, eval loss, eval accuracy
        '''
        data = {"test_loss": [tl for tl in self.monitor_loss]}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
        
    def set_dropout(self, b):
        for _,e in enumerate(self.modules):
            if hasattr(e, 'training'):
                e.set_training(b)
        
    def backward(self):
        grad = self.d_loss(self.output, self.target)
        for _,m in reversed (list(enumerate(self.modules))):
            grad = m.backward(grad)

    # call the update weights and bias of each linear layers
    def update(self,eta):
        for _,n in enumerate(self.modules):
            if (n.param() != []):
                n.update_weights(eta)
# --------------------------------------------------------------#
'''
Change the formula of the update of bias and weights according to Gradient Descent
(GD).
This enables to pass saddle points and local minima more easily
'''
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

'''
Change the formula of the update of bias and weights according to Adam optimization
This enables to pass saddle points and local minima more easily
'''  
class optimizer_Adam:
    def __init__(self, network):
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {} 
        for i,n in enumerate(network.modules):
            if (n.param() != []):
                self.m_w[i] = 0
                self.v_w[i] = 0
                self.m_b[i] = 0
                self.v_b[i] = 0
                
    def step(self, sequential, eta, epsilon, beta1, beta2):
        for i,n in enumerate(sequential.modules):
            if (n.param() != []):
                self.m_w[i] = self.m_w[i]*beta1 + (1-beta1)*n.dl_dw
                self.m_b[i] = self.m_b[i]*beta1 + (1-beta1)*n.dl_ds
                self.v_w[i] = self.v_w[i]*beta2 + (1-beta2)*n.dl_dw**2
                self.v_b[i] = self.v_b[i]*beta2 + (1-beta2)*n.dl_ds**2
                
                m_w_t  = self.m_w[i]/(1-beta1)
                m_b_t  = self.m_b[i]/(1-beta1)
                
                v_w_t = self.v_w[i]/(1-beta2)
                v_b_t = self.v_b[i]/(1-beta2)
                
                n.weights = n.weights - eta*m_w_t/(v_w_t**(0.5)+epsilon)
                n.bias = n.bias - eta*m_b_t/(v_b_t**(0.5)+epsilon)

#-----------------------------------------------------------------------------#
                
#------------------------------- INITIALIZATON -------------------------------#

# Generate sets
train_input, test_input, train_target, test_target = g.generate_sets()
# Turn train_target into one hot labels
train_labels = utility.one_hot(train_target,2)
# Create a network with loss = MSE
network = Sequential(train_labels, utility.d_MSE_loss, utility.MSE_loss)

#-----------------------------------------------------------------------------#

#---------------------------------- NETWORK ----------------------------------#
'''
Creation of the network by adding the different layers.
Here, composed of three hidden layers of size 25 in addition to input and output
layers of size 2.
Dropout is not mandatory, but added to test its functionnality -> it does not 
change much the result
Activation functions are sigmoid because ReLU could cause constant loss. However,
this does work with the first two layers having ReLU.
'''  
network.add(Linear(2,25))
network.add(Dropout(0.5,25))
network.add(Sigmoid())
network.add(Linear(25,25))
network.add(Sigmoid())
network.add(Linear(25,25))
network.add(Sigmoid())
network.add(Linear(25,2))
network.add(Sigmoid())
SGD = optimizer_SGD(network)
Adam = optimizer_Adam(network)
#-----------------------------------------------------------------------------#

#---------------------------------- TRAINING ---------------------------------#
NB_EPOCH = 100
accuracy = t.empty(NB_EPOCH) # keep track of accuracy evolution
for i in range(NB_EPOCH):
    network.forward(train_input)
    network.display()
    network.backward()
    
    #three different ways to update weights
    
    #SGD.step(network, 5e-3, gamma = 0.7)
    #network.update(5e-1)
    Adam.step(network, 5e-2, 1e-3, 0.5, 0.6)
    k = utility.calculate_error(network.output,train_target)
    accuracy[i] = (1000-k)/1000
    print('[epoch {:d}] loss: {:0.5f} error: {} accuracy: {}'.format(i+1, network.monitor_loss[i], k, accuracy[i]*100))
print("NB error for train: {}".format(utility.calculate_error(network.output,train_target)))
#-----------------------------------------------------------------------------#

#----------------------------------- TESTING ---------------------------------#
# Deactivate dropout for testing
network.set_dropout(False)
network.forward(test_input)
print("NB error for test: {}".format(utility.calculate_error(network.output,test_target)))
#-----------------------------------------------------------------------------#

#----------------------------------- PLOTTING --------------------------------#
with open('accuracy_ours_v2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(accuracy.numpy(), f)

with open('loss_ours_v2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(np.array(network.monitor_loss), f)
    
#pytorch framework
framework_pytorch.__main__()
pl.__main__()

