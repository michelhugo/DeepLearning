import numpy as np
import torch as t
import math

def generate_sets():
    train_input = t.rand(1000,2)
    test_input = t.rand(1000,2)
    train_target, test_target = t.empty(1000,1),t.empty(1000,1)
    for i,a in enumerate(train_input[:]):
        r = 1/math.sqrt(2*math.pi)
        dist = a[0]**2 + a[1]**2
        dist = math.sqrt(dist)
        if dist <= r:
            train_target[i] = 1
        else:
            train_target[i] = 0
            
    for i,a in enumerate(test_input[:]):
        r = 1/math.sqrt(2*math.pi)
        dist = a[0]**2 + a[1]**2
        dist = math.sqrt(dist)
        if dist <= r:
            test_target[i] = 1
        else:
            test_target[i] = 0
    
    return train_input, train_target, test_input, test_target

def generate_sets_r():
    train_input = t.rand(1000,2)
    test_input = t.rand(1000,2)
    dist = t.empty(1000,1)
    dist_ = t.empty(1000,1)
    train_target, test_target = t.empty(1000,1),t.empty(1000,1)
    
    for i,a in enumerate(train_input[:]):
        r = 1/math.sqrt(2*math.pi)
        dist[i] = a[0]**2 + a[1]**2
        dist[i] = math.sqrt(dist[i])
        if dist[i] <= r:
            train_target[i] = 1
        else:
            train_target[i] = 0
            
    for i,a in enumerate(test_input[:]):
        r = 1/math.sqrt(2*math.pi)
        dist_[i] = a[0]**2 + a[1]**2
        dist_[i] = math.sqrt(dist_[i])
        if dist_[i] <= r:
            test_target[i] = 1
        else:
            test_target[i] = 0
    train_input = t.cat((train_input,dist),dim=1)
    test_input = t.cat((test_input,dist_),dim=1)
    return train_input, train_target, test_input, test_target

def generate_sets_plus():
    train_input = t.rand(1000,2)
    test_input = t.rand(1000,2)
    train_target, test_target = t.empty(1000,1),t.empty(1000,1)
    for i,a in enumerate(train_input[:]):
        r = 1.5/math.sqrt(2*math.pi)
        dist = a[0]**2 + a[1]**2
        dist = math.sqrt(dist)
        if dist <= r:
            train_target[i] = 1
        else:
            train_target[i] = 0
            
    for i,a in enumerate(test_input[:]):
        r = 1.5/math.sqrt(2*math.pi)
        dist = a[0]**2 + a[1]**2
        dist = math.sqrt(dist)
        if dist <= r:
            test_target[i] = 1
        else:
            test_target[i] = 0
    
    return train_input, train_target, test_input, test_target

def generate_sets_g(nb_train = 1000, nb_test = 1000):
    # data
    train_set = t.Tensor(nb_train, 2).uniform_(0, 1)
    test_set = t.Tensor(nb_test, 2).uniform_(0, 1)
    # labels
    train_target = train_set.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().add(1).div(2).long()
    test_target = test_set.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().add(1).div(2).long()
    
    return train_set, train_target, test_set, test_target  
 