"""
All models and architectures

"""

###############################################################################
                            # - PARSING - #
###############################################################################
import argparse
parser = argparse.ArgumentParser(description='Run best models to compare \
                                 2 MNIST digit using 2 different architectures')
parser.add_argument('-l','--loss',action = 'store_true', help ='Display the final losses for each case')
parser.add_argument('-m','--model_large', action = 'store_true', help ='Choose large model for comparison over the small one')
parser.add_argument('-w','--weight_sharing', action = 'store_false', help = 'Disable weight sharing in 2-models architecture')
parser.add_argument('-b','--baseline', action = 'store_true', help = 'Display the baseline results')
parser.add_argument('-r','--learning_rate', action = 'store', type=float, default=1e-3,help = 'Change the value of the learning rate (default: 1e-3)')
parser.add_argument('-p','--epochs', action = 'store', type=int, default=25, help = 'Number of epochs that will be used to train the model')
parser.add_argument('-n','--runs', action = 'store', type=int, default=10, help = 'Number of runs to try the architecture')
args = parser.parse_args()

###############################################################################
                            # - IMPORT - #
###############################################################################
import torch as t
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from dlc_practical_prologue import generate_pair_sets
import time

###############################################################################
                            # - CONSTANTS - #
###############################################################################

#seed to retrieve the same results
t.manual_seed(0)
                          
NB_PAIRS = 1000
NB_EPOCHS = args.epochs
MINI_BATCH_SIZE = 100
NB_RUN = args.runs

#fetch data
train_input,train_target,train_classes,test_input,test_target,test_classes \
= generate_pair_sets(NB_PAIRS)

#wrap both inputs so that the gradients can be computed
train_input = Variable(train_input.float(),requires_grad=True)
test_input = Variable(test_input.float(),requires_grad=True)

#convert string learning rate to float to be used by optimizer
lr_ = args.learning_rate
###############################################################################
                            # - NETWORKS - #
###############################################################################

# Baseline model that perform classification of a 14x14 MNIST image
# Output is of size 10x1, containing the 10 classes power
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        #1x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,10)
        self.b1 = nn.BatchNorm1d(128)
        self.m = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 64)))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.m(self.fc3(x))
        return x

# Model used in the 2nd architecture: consists of 4 hidden layers, no convolution
# Input of size 2x10x1 (power for pair of digits) and output of size 2x1
class Comparison_fc_large(nn.Module):
    def __init__(self):
        super(Comparison_fc_large, self).__init__()
        self.fc1 = nn.Linear(20, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, 160)
        self.fc4 = nn.Linear(160, 80)
        self.fc5 = nn.Linear(80, 2)
        self.m = nn.Softplus()
        
    def forward(self, x):
        x = F.relu(x.view(-1,20))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.m(self.fc5(x))
        return x
    
# Model used in the 2nd architecture: consists of 2 hidden layers, no convolution
# Input of size 2x10x1 (power for pair of digits) and output of size 2x1
class Comparison_fc_small(nn.Module):
    def __init__(self):  
        super(Comparison_fc_small,self).__init__()
        self.fc1 = nn.Linear(20,40)
        self.fc2 = nn.Linear(40,80)
        self.fc3 = nn.Linear(80,2)
        self.m = nn.Softplus()
        
    def forward(self,x):
        x = F.relu(x.view(-1,20))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.m(self.fc3(x))
        return x

###############################################################################
                            # - ERROR COMPUTATION - #
###############################################################################

# Compute errors for the classification model         
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

# Compute errors for the comparison model
def compute_error_comparison (input,target):
    # input of size Nx2
    # target of size N
    
    error = 0
    for k in range(input.size(0)):
        _,n = t.max(input[k],0)
        if n != target[k]:
            error = error + 1
    return error

def compute_error_baseline(input,target):
    error = 0
    for k in range(input.size(0)):
        _,n1 = t.max(input[k,0],0)
        _,n2 = t.max(input[k,1],0)
        r = 0
        if n1<=n2:
            r = 1
        if r != target[k]:
            error = error + 1
    return error
###############################################################################
                            # - TRAINING & TESTING- #
###############################################################################

# Perform the training of the full architecture, with weight sharing
def training_ws(model_classification,model_comparison):
    criterion = nn.CrossEntropyLoss()
    optimizer_cl = t.optim.Adam(model_classification.parameters(),lr=lr_)
    optimizer_co = t.optim.Adam(model_comparison.parameters(),lr=lr_)
    input = train_input
    target_cl = train_classes
    target_co = train_target
    
    for e in range(NB_EPOCHS):
        sum_loss_classification = 0
        sum_loss_comparison = 0
        model_comparison.zero_grad()
        model_classification.zero_grad()
        for b in range(0,input.size(0),MINI_BATCH_SIZE):
            sum_loss = 0
            input_co = t.empty(MINI_BATCH_SIZE,2,10)
            #model_comparison.zero_grad()
            #model_classification.zero_grad()
            for i in range(2):
                output_cl = model_classification(input[b:b+MINI_BATCH_SIZE,i].view(100,1,14,14)) 
                loss_cl = criterion (output_cl,target_cl[b:b+MINI_BATCH_SIZE,i].long())
                input_co[:,i,:] = output_cl
                sum_loss = sum_loss + loss_cl
                
                
            # Update parameters after backward pass
            sum_loss_classification = sum_loss_classification + sum_loss.item()
            output_co = model_comparison(input_co)
            loss_co = criterion(output_co,target_co[b:b+MINI_BATCH_SIZE].long())
            sum_loss = sum_loss * 0.8 + loss_co * 0.2
            sum_loss_comparison = sum_loss_comparison + loss_co.item()
            sum_loss.backward()
            optimizer_co.step()
            optimizer_cl.step()
        if (args.loss):            
            print("\tEpoch no {} :\n\t\tClassification loss = {:0.4f}\n\t\tComparison loss = {:0.4f}"
                  .format(e+1,sum_loss_classification,sum_loss_comparison))

# Perform the training of the full architecture, without weight sharing
def training_wo(model_classification,model_comparison):
    criterion = nn.CrossEntropyLoss()
    optimizer_cl = t.optim.Adam(model_classification.parameters(),lr=lr_)
    optimizer_co = t.optim.Adam(model_comparison.parameters(),lr=lr_)
    input = train_input
    target_cl = train_classes
    target_co = train_target
    
    for e in range(NB_EPOCHS):
        sum_loss_classification = 0
        sum_loss_comparison = 0
        model_comparison.zero_grad()
        model_classification.zero_grad()
        for b in range(0,input.size(0),MINI_BATCH_SIZE):
            sum_loss = 0
            input_co = t.empty(MINI_BATCH_SIZE,2,10)
            
            # To disable weight sharing > make backward + optim_step for classification in the loop below
            for i in range(2):

                output_cl = model_classification(input[b:b+MINI_BATCH_SIZE,i].view(100,1,14,14)) 
                loss_cl = criterion (output_cl,target_cl[b:b+MINI_BATCH_SIZE,i].long())
                input_co[:,i,:] = output_cl
                sum_loss = sum_loss + loss_cl
                
                loss_cl.backward(retain_graph = True)
                optimizer_cl.step()
            
            # Update parameters after backward pass
            sum_loss_classification = sum_loss_classification + sum_loss.item()
            output_co = model_comparison(input_co)
            loss_co = criterion(output_co,target_co[b:b+MINI_BATCH_SIZE].long())
            sum_loss = sum_loss * 0.8 + loss_co * 0.2
            sum_loss_comparison = sum_loss_comparison + loss_co.item()
            sum_loss.backward()
            optimizer_co.step()
        if (args.loss):
            print("\tEpoch no {} :\n\t\tClassification loss = {:0.4f}\n\t\tComparison loss = {:0.4f}"
                  .format(e+1,sum_loss_classification,sum_loss_comparison))

# Training function for the baseline model
def training_baseline(baseline):
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(baseline.parameters(),lr=lr_)
    input = train_input
    target = train_classes
    for e in range(NB_EPOCHS):
        baseline.zero_grad()
        for b in range(0,input.size(0),MINI_BATCH_SIZE):     
            for i in range(2):

                output = baseline(input[b:b+MINI_BATCH_SIZE,i].view(100,1,14,14)) 
                loss = criterion (output,target[b:b+MINI_BATCH_SIZE,i].long())
                
                loss.backward(retain_graph = True)
                optimizer.step()

# Once trained, the full architecture is used on testing and training sets
# Both errors are returned
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
    print("\033[1;35;40mError in {}: \n\tClassification = {:0.3f}% \n\tComparison = {:0.3f}%\033[0m"
          .format(s,err1*100,err2*100))
    return err1, err2

# Test the baseline model only if the option is chosen
def test_baseline(baseline, input, target, train=True):
    output = t.empty(NB_PAIRS,2,10)
    s = "test"
    if train:
        s = "train"
    for i in range(2):
        output[:,i,:] = baseline(input[:,i].view(NB_PAIRS,1,14,14))
    err = compute_error_baseline(output,target)/NB_PAIRS
    print("\033[1;36;40mError in baseline {}: {:0.3f}%\033[0m".format(s,err*100))

###############################################################################
                            # - MAIN - #
###############################################################################
m="Small model"
if args.model_large:
    m = "Large model"

# Display all the parameters that will be used
print("\033[0;37;42m---------------------------------------------------------------\033[0m")
print("Here are the parameters used for this run:")
print("Learning rate: {}\nNumber of epochs: {}\nNumber of runs: {}\nWeight sharing: {}\nModel used for comparison: {}"
      .format(lr_, NB_EPOCHS, NB_RUN, args.weight_sharing,m))                            
print("\033[0;37;42m---------------------------------------------------------------\033[0m")
# Store some parameters over runs to make mean and standard deviation
train_class = t.empty(NB_RUN)
train_comp = t.empty(NB_RUN)
test_class = t.empty(NB_RUN)
test_comp = t.empty(NB_RUN)

training_time = t.empty(NB_RUN)

for i in range(NB_RUN):
    #classification model for 2-model architecture
    model_cl = Classification()
    # if large comparison model, otherwise small (3 layers)
    if (args.model_large):
        model_co = Comparison_fc_large()
    else:
        model_co = Comparison_fc_small()
    print("\033[0;37;45mRUN NO {}\033[0m".format(i+1))
    
    # if baseline model is selected
    if args.baseline:
        baseline = Classification()
        training_baseline(baseline)
        test_baseline(baseline, train_input, train_target)
        test_baseline(baseline, test_input, test_target, False)
    #Compute training time
    start_time = time.perf_counter()
    if args.weight_sharing:
        training_ws(model_cl,model_co)
    else:
        training_wo(model_cl,model_co)
    stop_time = time.perf_counter()
    
    training_time[i] = stop_time - start_time

    #test the trained architecture on the whole training/testing sets
    e1, e2 = test_models(model_cl, model_co, train_input, train_classes, train_target)
    e3, e4 = test_models(model_cl, model_co, test_input, test_classes, test_target, False)
    #store the errors to use afterward
    train_class[i] = (e1)
    train_comp[i] = (e2)
    test_class[i] = (e3)
    test_comp[i] = (e4)

# Display mean of errors over NB_RUN and standard deviations as well as average training time
print("\033[1;31;40mFinal error for train batch : {:0.2f}\u00B1{:0.4f}%\033[0m".format(t.mean(train_comp)*100,t.std(train_comp)*100))
print("\033[1;31;40mFinal error for test batch : {:0.2f}\u00B1{:0.4f}%\033[0m".format(t.mean(test_comp)*100,t.std(test_comp)*100))

print('\033[1;31;40mAverage training time: {:f}s\033[0m'.format(t.mean(training_time)))