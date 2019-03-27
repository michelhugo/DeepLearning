import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim
import dlc_practical_prologue as prologue



train_input, train_target, train_class,test_input, test_target,test_class = \
    prologue.generate_pair_sets(1000)
    

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)
mini_batch_size = 100



######################################################################
class Net(nn.Module):
    def __init__(self, nb_hidden):
        super(Net, self).__init__() 
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32,64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
    
    
    def forward(self, x): 
    
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = x.view(-1,256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
       
        return x
    
    
######################################################################

def train_model(model, train_input, train_target, mini_batch_size):
    criterion = nn.CrossEntropyLoss() #set the criterion with 
    eta = 1e-3 #learning rate 
    optimizer = optim.SGD(model.parameters(), lr = eta)
    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output,train_target.narrow(0, b, mini_batch_size).long())
            model.zero_grad()
            loss.backward()
            optimizer.step()

            
            sum_loss = sum_loss + loss.item()
            
            #Update the weights
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
                #substract gradient to param
        print(e, sum_loss)
       

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        
        for k in range(mini_batch_size):
            if target.data[k+b] != predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors



######################################################################


for k in range(10):
    model = Net(200)
    train_model(model, train_input, train_target.float(), mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))