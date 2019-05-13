"""
The current code was widely inspired from the book of Michael Nielsen 
(c.f. http://neuralnetworksanddeeplearning.com/chap1.html). The code can
be accessed on the github repo: https://github.com/mnielsen/neural-networks-and-deep-learning

A network can be defined by the size of its layers and the cost function 
associated to it (e.g. Network(sizes = [2, 8, 4, 2], cost=CrossEntropyCost))

"""

#### Imports
import json
import random
import sys
import numpy as np
import math

import torch



#### Loss: quadratic and cross-entropy
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        '''
        Return the cost associated with an output 'a' and desired output 'y'
        '''
        return 0.5*np.linalg.norm(a-y)**2 #TODO: make sure we use the same def as in our network in order to compare the cost
        #return np.linalg.norm(a-y)**2
        
    @staticmethod
    def delta(z, a, y):
        '''
        Return the error delta from the output layer
        '''
        return (a-y) * sigmoid_prime(z) #depending on the definition of the mse
        #return 2/1000*(a-y) * sigmoid_prime(z)
        #return 2*(a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost
        
        # Aim: recording the cost (train, eval) and accuracy
        self.record_train_cost = []
        self.record_evaluation_cost = []
        self.record_train_accuracy = []
        self.record_evaluation_accuracy = []

    def default_weight_initializer(self):
        '''
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.
        '''
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        '''
        Initialize the weights and biases using a Gaussian distribution 
        with mean 0 and standard deviation 1.
        '''
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        '''
        Return the output of the network if 'a' is input.
        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        '''
        Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples (x, y)
        representing the training inputs and the desired outputs.
        '''
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] #xrange
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                self.record_train_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                self.record_train_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                self.record_evaluation_cost.append(cost) #record the cost/loss
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data) #de base convert=False
                evaluation_accuracy.append(accuracy)
                self.record_evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        '''
        Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        Return a tuple (nabla_b, nabla_w) representing the
        gradient for the cost function C_x.  
        nabla_b: dl_db 
        nabla_w: dl_dw
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # forward pass
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers): 
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        '''
        Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.
        '''
        
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum((x == y).astype(int) for (x, y) in results) #I replaced int(x==y) to (x==y).astype(int)

    def total_cost(self, data, lmbda, convert=False):
        '''
        Return the total cost for the data set 'data'.
        '''
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file 'filename'."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    
    # Add this function to save loss/cost
    def save_loss(self, filename):
        '''
        Save the network loss
        '''
        data = {"eval_cost": [c.tolist() for c in self.record_evaluation_cost]}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
    
    def save_params(self, filename):
        '''
        Save the network train loss, train accuracy, eval loss, eval accuracy
        '''
        data = {"train_cost": [tc.tolist() for tc in self.record_train_cost],
                "eval_cost": [ec.tolist() for ec in self.record_evaluation_cost],
                "train_accuracy": [ta.tolist() for ta in self.record_train_accuracy],
                "test_accuracy": [ea.tolist() for ea in self.record_evaluation_accuracy]}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
                

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    '''
    Return a 2-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    '''
    
    j = j.astype(int) #add this line
    e = np.zeros((2, 1)) # change here into 2 to monitor the cost!
    e[j] = 1.0
    return e

def sigmoid(z):
    '''Sigmoid activation function'''
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    '''Derivative of the sigmoid activation function'''
    return sigmoid(z)*(1-sigmoid(z))



##################################
def generate_sets(nb_train = 1000, nb_test = 1000):
    # data
    train_set = torch.Tensor(nb_train, 2).uniform_(0, 1)
    test_set = torch.Tensor(nb_test, 2).uniform_(0, 1)
    # labels
    train_target = train_set.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().add(1).div(2).long()
    test_target = test_set.pow(2).sum(1).sub(1 / math.sqrt(2 * math.pi)).sign().add(1).div(2).long()
    
    return train_set, test_set, train_target, test_target  
 
###################################


########## MAIN ###################
train_input, test_input, train_target, test_target = generate_sets()

np_train_input = train_input.numpy()
np_test_input = test_input.numpy()
np_train_target = train_target.numpy()
np_test_target = test_target.numpy()


# The input is a list of np.array of the shape:
# [array(784, 1), array(10, 1)] > column vector
# Note: one-hot encoding format
nb_labels = 2
train_labels = (np.arange(nb_labels) == np_train_target[:, None]).astype(np.float32)
test_labels = (np.arange(nb_labels) == np_test_target[:, None]).astype(np.float32)

# Format the input data
training_input = [np.reshape(x, (2, 1)) for x in np_train_input]
np_train_i = np.array(training_input)
training_target = [np.reshape(y, (2, 1)) for y in train_labels]
np_train_t = np.array(training_target)

train_zip = zip(np_train_i, np_train_t)
train_data = list(train_zip)

# Format the output data
testing_input = [np.reshape(x,  (2, 1)) for x in np_test_input]
np_test_i = np.array(testing_input)
testing_target = [np.reshape(y, (2, 1)) for y in test_labels]
np_test_t = np.array(testing_target)

test_zip = zip(np_test_i, np_test_t)
test_data = list(test_zip)


net = Network(sizes = [2, 25, 25, 25, 2], cost=QuadraticCost)

net.SGD(train_data, 100, 10, 0.5, evaluation_data=test_data, 
        monitor_training_cost=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_evaluation_accuracy=True)

#net.save_params('100_epochs_2.json')
#net.save_loss('test.json')
#net.save_loss('test.csv')

#for i in range(20):
    #net.SGD(train_data, 30, 10, 0.5, evaluation_data=test_data, 
            #monitor_evaluation_cost=True,
            #monitor_evaluation_accuracy=True)


#net.SGD(train_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
#net.SGD(train_data, 30, 10, 0.5)








