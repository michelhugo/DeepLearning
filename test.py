import torch as t
###############################################################################
                            # - IMPORT - #
###############################################################################
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

###############################################################################
                            # - CONSTANTS - #
###############################################################################
                            
NB_PAIRS = 1000
NB_EPOCHS = 25
MINI_BATCH_SIZE = 100 # seems to be quite optimal (see lesson 5.2)
NB_RUN = 10


train_input,train_target,train_classes,test_input,test_target,test_classes \
= prologue.generate_pair_sets(NB_PAIRS)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

###############################################################################
                            # - NETWORKS - #
###############################################################################

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        #1x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256,10)
        self.b1 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = self.b1(x)
        x = F.relu(self.fc3(x))
        #batchnorm here does not change much
        #dropout neither
        x = self.fc2(x)
        return x

class Comparison(nn.Module):
    def __init__(self):
        super(Comparison,self).__init__()
        self.fc1 = nn.Linear(20,64)
        self.fc2 = nn.Linear(64,1)
        self.m = nn.Sigmoid()
        
    def forward(self,x):
        x = x.view(-1,20)
        x = F.relu(self.fc1(x))
        x = self.m(self.fc2(x))
        return x
    
    
###############################################################################
                            # - ERROR COMPUTATION - #
###############################################################################
                            
def compute_error_classification (input, target):
    # input of size Nx2x10
    # target of size Nx2
    # max error N*2 = 2N
    error = 0
    for k in range(input.size(0)):
        for i in range(2):
            _,n = t.max(input[k,i],0)
            if n != target[k,i]:
                error = error + 1
    return error

def compute_error_comparison (input,target):
    # input and target of size N
    
    error = 0
    for k in range(input.size(0)):
        if abs(input[k] - target[k].float())>0.1: # how to compare 0 with 1e-7 ?
            error = error + 1
    return error

###############################################################################
                            # - TRAINING & TESTING- #
###############################################################################
                            
def training(model_classification,model_comparison):
    lr_ = 1e-3
    criterion_cl = nn.CrossEntropyLoss()
    criterion_co = nn.BCELoss()
    optimizer_cl = t.optim.Adam(model_classification.parameters(),lr=lr_)
    optimizer_co = t.optim.Adam(model_comparison.parameters(),lr=10*lr_)
    input = train_input
    target_cl = train_classes
    target_co = train_target
    
    for e in range(NB_EPOCHS):
        sum_loss_classification = 0
        sum_loss_comparison = 0
        
        for b in range(0,input.size(0),MINI_BATCH_SIZE):
            input_co = t.empty(MINI_BATCH_SIZE,2,10)
            for i in range(2):
                output_cl = model_classification(input[b:b+MINI_BATCH_SIZE,i,:,:].view(100,1,14,14))
                loss_cl = criterion_cl (output_cl,target_cl[b:b+MINI_BATCH_SIZE,i].long())
                model_classification.zero_grad()
                loss_cl.backward(retain_graph=True)
                sum_loss_classification = sum_loss_classification + loss_cl.item()
                # Update parameters after backward pass
                optimizer_cl.step()
                input_co[:,i,:] = output_cl
            output_co = model_comparison(input_co)
            loss_co = criterion_co(output_co,target_co[b:b+MINI_BATCH_SIZE].float())
            model_comparison.zero_grad()
            loss_co.backward()#retain_graph=True
            sum_loss_comparison = sum_loss_comparison + loss_co.item()
            optimizer_co.step()
        print("Epoch no {} : \nClassification loss = {:0.4f}\nComparison loss = {:0.4f}".format(e+1,sum_loss_classification,sum_loss_comparison))

def test_models(model_cl, model_co, input, classes, target, train=True):
    output_cl = t.empty(NB_PAIRS,2,10)
    for i in range(2):
        output_cl[:,i,:] = model_cl(input[:,i].view(NB_PAIRS,1,14,14))
    output_co = model_co(output_cl)
    if (train):
        s = "training"
    else:
        s = "testing"
    err1 = compute_error_classification(output_cl,classes)/2/NB_PAIRS
    err2 = compute_error_comparison(output_co, target)/NB_PAIRS
    print('\x1b[3;37;41m'+"Error in {}: \nClassification = {:0.3f}% \nComparison = {:0.3f}%".format(
          s,err1*100
          ,err2*100)+'\x1b[0m')
    return err1, err2
###############################################################################
                            # - MAIN - #
###############################################################################
train_class = t.empty(NB_RUN)
train_comp = t.empty(NB_RUN)
test_class = t.empty(NB_RUN)
test_comp = t.empty(NB_RUN)
for i in range(NB_RUN):
    model_cl = Classification()
    model_co= Comparison()
    print("RUN NO {}".format(i+1))
    training(model_cl,model_co)

    e1, e2 = test_models(model_cl, model_co, train_input, train_classes, train_target)
    e3, e4 = test_models(model_cl, model_co, test_input, test_classes, test_target, False)
    train_class[i] = (e1)
    train_comp[i] = (e2)
    test_class[i] = (e3)
    test_comp[i] = (e4)

print("Final error for train batch : {:0.2f}\u00B1{:0.4f}".format(t.mean(train_comp)*100,t.std(train_comp)*100))
print("Final error for test batch : {:0.2f}\u00B1{:0.4f}".format(t.mean(test_comp)*100,t.std(test_comp)*100))
