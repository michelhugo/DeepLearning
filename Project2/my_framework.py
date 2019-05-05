import math
import torch

########## Generic class ##########

class Module(object):
    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        return[]
        
###################################

########## Functions ##############
        
def relu(x):
    output = torch.clone(x)
    output[x <= 0] = 0
    return output
    
def drelu(x):
    output = torch.clone(x)
    output[x > 0] = 1
    output[x <= 0] = 0
    return output

def tanh(x):
    return (x.exp() - x.mul(-1).exp()) * ((x.exp() + x.mul(-1).exp()).pow(-1))

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def l2_norm(x, y):
    y_hat = Module.forward(x)
    return ((y_hat - y).pow(2)).sum().item()

def lossMSE(x, y):
    '''
    Compute the mean squared error (MSE) between the x and y input tensors
    '''
    return ((x - y).pow(2)).mean().item()

def dlossMSE(x, y):
    return (2 * (x - y)).mean().item()


def linear(input, weights, bias):
    '''
    Compute the forward pass.
    The linear computation is performed as follow: output = input * weights + bias
    with respective shapes of:
        - input: [N x in_features]
        - weights: [out_features, in_features]
        - bias: [out_features x 1]
        - output: [N x out_features]
    where: 
        - N: number of samples
        - in_features: number of features of the input layer
        - out_features: number of features of the ouput layer   
    '''
    # Or input * weights + bias
    # Or input.matmul(weights) + bias #better as we're dealing with tensors
    # Or input @ weights + bias
    return (input.matmul(weights) + bias) 
    
    
    
# def sequential()
    




###################################

########## Sub-classes ############
class ReLU(Module):
    '''
    Apply the ReLU activation function
    '''
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return relu(input, inplace = self.inplace)

class Tanh(Module):
    '''
    Apply the Tanh activation function
    '''
    def __init__(self, inplace=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return tanh(input, inplace = self.inplace)
    

class Linear():
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.Tensor(in_features, out_features) # What about params at this point?
        self.bias = torch.Tensor(out_features)
    
    def forward(self, input):
        return linear(input, self.weights, self.bias)
   
     
# class Sequential():
    
    
    
###################################



########## Train - Test sets ######
    
def generate_sets(nb_train = 1000, nb_test = 1000):
    # data
    train_set = torch.Tensor(nb_train, 2).uniform_(0, 1)
    test_set = torch.Tensor(nb_test, 2).uniform_(0, 1)
    # labels
    train_target = train_set.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().long()
    test_target = test_set.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().long()
    
    return train_set, test_set, train_target, test_target  
 
###################################


########## MAIN ###################
train_, test_, train_labels, test_labels = generate_sets()
print(train_.shape)
print(test_.shape)
print(train_labels.shape)
print(test_labels.shape)


###################################




