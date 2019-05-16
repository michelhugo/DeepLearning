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
import matplotlib.pyplot as plt
import json

t.set_grad_enabled(False)
t.manual_seed(0)


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

def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T

def cross_entropy(model, y_pred, y_train, lam=1e-3):
    m = y_pred.shape[0]

    prob = softmax(y_pred)
    log_like = -np.log(prob[range(m), y_train])

    data_loss = np.sum(log_like) / m
    reg_loss = .5 * lam * np.sum(model * model)

    return data_loss + reg_loss


def dcross_entropy(y_pred, y_train):
    m = y_pred.shape[0]

    grad_y = util.softmax(y_pred)
    grad_y[range(m), y_train] -= 1.
    grad_y /= m

    return grad_y

def one_hot(labels, nb_labels):
    '''
    input labels can be either np array or tensors
    output is a torch.tensor
    '''
    if (type(labels) == np.ndarray):
        h_labels = (np.arange(nb_labels) == labels[:, None]).astype(np.float32)

    elif (type(labels) == t.Tensor):
        h_labels = labels.numpy()
        h_labels = (np.arange(nb_labels) == h_labels[:, None]).astype(np.float32)
        
    else:
        raise ValueError('The input type must be either numpy.ndarray or torch.Tensor')
    
    return t.Tensor(h_labels).view(labels.size(0),nb_labels)
    
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
        return self.activation(s)
        
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
        return self.activation(s)
        
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
        return self.activation(s)
        
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
        return t.mm(self.input,self.weights) + self.bias #wt + b
    
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
class Dropout(Module):
    def __init__(self, p, input_size, seed=0):
        self.p = p
        self.generator = np.random.RandomState(seed)
        self.activation = self.generator.binomial(size=input_size, n=1, p=1-p)
        self.activation = t.from_numpy(self.activation).float()
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
        
    # remove last element of vector
    def remove(self):
        self.modules = self.modules[0:-2]

    def forward(self, input):
        next_in = input
        for i,_ in enumerate(self.modules):
                next_in = self.modules[i].forward(next_in)
        
        self.output = next_in
        
    def display(self):
        print(self.loss(self.output,self.target))
        #print(self.output)
        self.monitor_loss.append(self.loss(self.output,self.target))
        
    def backward(self):
        grad = self.d_loss(self.output, self.target)
        for _,m in reversed (list(enumerate(self.modules))):
            grad = m.backward(grad)

    def update(self,eta):
        for _,n in enumerate(self.modules):
            if (n.param() != []):
                n.update_weights(eta)
                
    def save_params(self, filename):
        '''
        Save the network train loss, train accuracy, eval loss, eval accuracy
        '''
        data = {"test_loss": [tl for tl in self.monitor_loss]}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
        
# --------------------------------------------------------------#
#pow of size nxC (number of classes)
def calculate_error_power(power,tar):
    _,ind = power.max(1)
    i = ind.clone().detach()
    diff = tar.long().view(tar.size(0))-i
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

#test_labels,train_labels = target_to_one_hot(test_target),target_to_one_hot(train_target)


#train_input, train_target, test_input, test_target = g.generate_sets_g()


def generate_sets(nb_train = 1000, nb_test = 1000):
    # data
    train_set = t.Tensor(nb_train, 2).uniform_(0, 1)
    test_set = t.Tensor(nb_test, 2).uniform_(0, 1)
    # labels
    train_target = train_set.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    test_target = test_set.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    
    return train_set, test_set, train_target, test_target  


train_input, test_input, train_target, test_target = generate_sets()
train_labels = one_hot(train_target,2)
network = Sequential(train_labels, d_MSE_loss, MSE_loss)


#network.add(Linear(2,8))
#network.add(Dropout(0.5,8))
#network.add(ReLU())
#network.add(Linear(8,16))
#network.add(Dropout(0.5,16))
#network.add(ReLU())
#network.add(Linear(16,1))
#network.add(Sigmoid())

network.add(Linear(2,25))
#network.add(Dropout(0.5,25))
network.add(Sigmoid())
network.add(Linear(25,25))
network.add(Sigmoid())
network.add(Linear(25,25))
network.add(Sigmoid())
network.add(Linear(25,2))
network.add(Sigmoid())
SGD = optimizer_SGD(network)
Adam = optimizer_Adam(network)
NB_EPOCH = 1000
accuracy = t.empty(NB_EPOCH)

for i in range(NB_EPOCH):
    network.forward(train_input)
    network.display()
    network.backward()
    
    #SGD.step(network, 5e-3, gamma = 0.7)
    #network.update(5e-1)
    Adam.step(network, 5e-2, 1e-4, 0.5, 0.99) #good shape: 0.5; 0.99
    #print(network.output)
    k = calculate_error_power(network.output, train_target)
    accuracy[i] = (1000-k)/1000 #100-k/10

### Save parameters ###
network.save_params('loss_hugo_v2.json')

#accuracies = {"test_loss": [acc for acc in accuracy]}
#f = open('accuracy_hugo_v2.json', 'w')
#json.dump(accuracies, f)
#f.close()

import pickle
with open('accuracy_hugo_v2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(accuracy.numpy(), f)
        

print("NB error for train: {}".format(calculate_error_power(network.output,train_target)))

#network.modules[1].set_training(False)
#network.modules[4].set_training(False)
#network.modules[7].set_training(False)
network.forward(test_input)
print("NB error for test: {}".format(calculate_error_power(network.output,test_target)))


plt.subplots(figsize=(12, 6))
plt.plot(accuracy.numpy(), label='Framework 1')
plt.xlabel('epochs')
plt.ylabel('Accuracy [%]')
plt.title('Accuracy evolution during the training')