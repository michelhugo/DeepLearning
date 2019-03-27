import torch as t
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

NB_PAIRS = 1000

train_input,train_target,train_classes,test_input,test_target,test_classes \
= prologue.generate_pair_sets(NB_PAIRS)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

NB_EPOCHS = 25

# To be defined:
MINI_BATCH_SIZE = 100 # seems to be quite optimal (see lesson 5.2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #1x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = F.relu(self.fc3(x))
        x = self.fc2(x)
        return x

def train_model(model, input, target, criterion, lr_):
    #input of size 1000x2x14x14
    #goal is 1000x2x10 -> 2000x1x14x14
    optimizer = t.optim.Adam(model.parameters(), lr=lr_)#,momentum=0.002,dampening = 0.002, weight_decay = 0.002)
    loss_table = []
    in_ = convert_to_in(input)
    tar_ = process_target(target)
    for e in range(NB_EPOCHS):
        sum_loss = 0
        for b in range(0,input.size(0),MINI_BATCH_SIZE):
            output = model(in_[b:b+MINI_BATCH_SIZE])
            loss = criterion (output,tar_[b:b+MINI_BATCH_SIZE].long())
            model.zero_grad()
            loss.backward()
            sum_loss = sum_loss + loss.item()
            # Update parameters after backward pass
            optimizer.step()
        loss_table.append(sum_loss)
        print("Epoch no {:d} -> loss = {:0.2f}".format(e+1,sum_loss))
        
def compute_error(model, input, target):
    in_ = convert_to_in(input)
    output = model(in_)
    output = process_output(output)
    error = 0
    for i in range(1000):
        if output[i].item() != target[i]:
            error = error + 1
    return error

def process_output(input):
    #input should be of size 2000x10
    out = t.empty(NB_PAIRS, dtype=t.int32)
    for i in range(NB_PAIRS):
        _,m1 = t.max(input[i],0)
        _,m2 = t.max(input[NB_PAIRS+i],0)
        if m1<=m2:
            out[i] = 1
        else:
            out[i] = 0
    return out

def process_target(target):
    tar1 = target[:,0]
    tar2 = target[:,1]
    tar_ = t.cat((tar1,tar2),0)
    return tar_


#convert to trainable data
def convert_to_in(input):
    r = t.zeros(2*NB_PAIRS,1,14,14)
    in1 = input[:,0,:,:].view(NB_PAIRS,1,14,14)
    in2 = input[:,1,:,:].view(NB_PAIRS,1,14,14)
    r[0:NB_PAIRS,:,:,:] = in1
    r[NB_PAIRS:2*NB_PAIRS,:,:,:] = in2
    return r
    
criterion = nn.CrossEntropyLoss() #-> better to use crossEntropy for classification !!
test_error = t.empty(10)
train_error = t.empty(10)
for e in range(10):
    model = Net()
    train_model(model,train_input,train_classes,criterion,1e-3)
    test_error[e] = compute_error(model,test_input,test_target)/NB_PAIRS*100
    train_error[e] = compute_error(model,train_input,train_target)/NB_PAIRS*100
    print("Error rate in testing: {}%".format(test_error[e]))
    print("Error rate in training: {}%".format(train_error[e]))
print("Mean error for training: {:0.2f}\u00B1{:0.2f}% and testing: {:0.2f}\u00B1{:0.2f}%".format(t.mean(train_error),t.std(train_error),t.mean(test_error),t.std(test_error)))