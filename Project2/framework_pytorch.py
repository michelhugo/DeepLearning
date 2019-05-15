# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:23:30 2019

@author: Gianni
"""

import numpy as np
import torch 
import math
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

torch.set_grad_enabled(False)
torch.manual_seed(0)

### Miscellaneous functions
def accuracy(y_predicted, y_actual):
    '''
    Accuracy of binary prediction with values 0, 1
    Convert both entry tensors to numpy
    '''
    if (type(y_predicted) == torch.Tensor):
        y_predicted = y_predicted.detach().numpy()
        y_actual = y_actual.detach().numpy()
        #Note: without detach > error message: Can't call numpy() on Variable that requires grad
        
    return np.sum(y_predicted == y_actual) / len(y_actual)

def one_hot(labels, nb_labels):
    '''
    input labels can be either np array or tensors
    output is a torch.tensor
    '''
    if (type(labels) == np.ndarray):
        h_labels = (np.arange(nb_labels) == labels[:, None]).astype(np.float32)

    elif (type(labels) == torch.Tensor):
        h_labels = labels.numpy()
        h_labels = (np.arange(nb_labels) == h_labels[:, None]).astype(np.float32)
        
    else:
        raise ValueError('The input type must be either numpy.ndarray or torch.Tensor')
    
    return torch.Tensor(h_labels) 


### PyTorch model with useful functions
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,25)
        self.fc2 = nn.Linear(25,25)
        self.fc3 = nn.Linear(25,25)
        self.fc4 = nn.Linear(25,2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


def train_model(model, train_input, h_train_target, train_target, mini_batch_size, monitor_params):
    '''
    h_train_target: one-hot encoding target. Required for MSELoss (TODO: moduler avec 'if')
    train_target: normal targets. Required to compute the error
    '''
    
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr = 1e-3)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    nb_epochs = 100
    
    loss_storage = []
    error_storage = []
    accuracy_storage = []
    
    for e in range(nb_epochs):
        sum_loss = 0
        sum_error = 0
        sum_acc = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            
            ### Compute class from the output ###
            _, predicted_classes = torch.max(output.data, 1)
            
            ### Compute loss ###
            loss = criterion(output, h_train_target.narrow(0, b, mini_batch_size))
            
            ### Compute train error ###
            nb_errors = 0
            for k in range(mini_batch_size):
                if train_target.data[b + k] != predicted_classes[k]:
                    nb_errors = nb_errors + 1
            
            ### Compute accuracy ### 
            acc = accuracy(predicted_classes, train_target.narrow(0, b, mini_batch_size))
            
            sum_loss += loss.item() # compute loss for each mini batch for 1 epoch
            sum_error += nb_errors
            sum_acc += acc # ok
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_storage.append(sum_loss)
        error_storage.append(sum_error)
        accuracy_storage.append(sum_acc)
                
        print('[epoch {:d}] loss: {:0.2f} error: {} accuracy: {}'.format(e+1, sum_loss, sum_error, sum_acc))
        
    
    if monitor_params:
        return loss_storage, error_storage, accuracy_storage
    
### Generate sets
def generate_sets(nb_train = 1000, nb_test = 1000):
    # data
    train_set = torch.Tensor(nb_train, 2).uniform_(0, 1)
    test_set = torch.Tensor(nb_test, 2).uniform_(0, 1)
    # labels
    train_target = train_set.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    test_target = test_set.pow(2).sum(1).sub(1 / (2 * math.pi)).sign().add(1).div(2).long()
    
    return train_set, test_set, train_target, test_target  


### Main
train_input, test_input, train_target, test_target = generate_sets()
train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

#Sanity check
print(train_input.shape)
print(train_target.shape)
print(test_input.shape)
print(test_target.shape)

train_target_h = one_hot(train_target, 2)
test_target_h = one_hot(test_target, 2)


mini_batch_size = 10

model = Net()
losses_, errors_, accuracies_ = train_model(model, train_input, train_target_h, train_target, mini_batch_size, True)
nb_test_errors = compute_nb_errors(model, test_input, test_target)
print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))

### Plots
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

f1 = ax1.plot(np.array(losses_)[:60], color='darkblue', label='Loss')

f2 = ax2.plot(np.array(accuracies_)[:60], color='crimson', label='Accuracy')

fs = f1 + f2
labs = [l.get_label() for l in fs]
ax1.legend(f1+f2, labs, loc='center right', fontsize=15)

ax1.set_xlabel('Epochs', fontsize=15)
ax1.set_ylabel('MSE loss', fontsize=15)
ax2.set_ylabel('Accuracy [%]', fontsize=15)

plt.title('Loss and accuracy monitoring - PyTorch', fontsize=25)


