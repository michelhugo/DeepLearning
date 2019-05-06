import numpy as np
import torch as t
import math

def generate_sets():
    train_input = -2*t.rand(1000,2)+1
    test_input = -2*t.rand(1000,2)+1
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