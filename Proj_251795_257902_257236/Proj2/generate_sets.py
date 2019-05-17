import torch as t
import math

'''
Generate the training and testing sets, as well as targets. They consist in pairs of points
in [0,1]^2
'''

def generate_sets(nb_train = 1000, nb_test = 1000):
    # data
    train_set = t.Tensor(nb_train, 2).uniform_(0, 1)
    test_set = t.Tensor(nb_test, 2).uniform_(0, 1)
    # labels
    train_target = train_set.pow(2).sum(1).sub(1 / 2 / math.pi).sign().sub(1).div(-2).long()
    test_target = test_set.pow(2).sum(1).sub(1 / 2 / math.pi).sign().sub(1).div(-2).long()
    return train_set, test_set, train_target, test_target
