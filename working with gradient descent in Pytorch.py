###                                                  GRADIENT DESCENT
# given a set of points there will several lines that can
# fit the data. The best line is represented as such where
# the loss fucntion (y_hat-y)^2 of all the points aka 
# Root mean square error or RMSE is the smallest  
# RMSE = √[ Σ(y_hat – y)^2 / n ]
# or

# setup virtual environment and install at command line 
# python -m venv my_env
# pip3 install Scikit-learn

from sklearn.metrics import mean_squared_error
from math import sqrt
y_actual = []
y_predicted = []
rmse = sqrt(mean_squared_error(y_actual, y_predicted))

# to find these optimal parameters for minimal loss 
# we need to take the Derivative of our RMSE 
# RMSE.backward()     ---see also .backward()----
# and
# my_parameters.grad  ---see also .grad----
# and swap in data values. The  values that give us the
# lowest RMSE is the best fit.
# in linear regression fittment and there fore predictions are dependent on 2 things
# the slope changeing and the intercept chaneging

# this process of swapping in values into the Derivative of our loss f(x) to get new batches of predictions 
# summarized by the RMSE calculation is called
#       GRADIENT DESCENT








####################################################################### GRADIENT DESCENT WITH 1 TERM ####################################################
############################################################################################################################################################
# in Pytorch all values must be tensors
# lets create a simple linear data set

#setup fucntions and data
################################
import torch
w = torch.tensor(-10,requires_grad = True)   # this is like the +b in y=mx+b
x = torch.tensor.arange(-3,3,0,1).view(-1,1) # range from -3 to 3 by increments of .1 view adds another dimension not sure why this is needed
f = 3*x                                      # with this f(x) we can geta guideline
y = f+.1*torch.randn(x.size())               # add random noise and get y values


import matplotlib.pyplot as plt             # plt.plot(x,y)
plt.plot( x.numpy,y.numpy())                # numpy changes tensors to arrays for plotting          

# set up loss/cost and learning rate
###############################
def forward(x):
    return w*x                              # a way to generate y_hat w will change by .1 in our fuction

def criterion(yhat,y):                      # a loss fuction that compares y valaues
    return torch.mean((yhat-y)**2)



lr = .1

# Loop it all together
################################
for epoch in range(4):                      # for every element of data multiply 10
    Yhat = forward(x)                       # but why do it 4 times?????
    loss = criterion(Yhat,y)
    loss.backward()                         #gives us the derivative of 
    #w.grad                                 #returns derivative at -10
    w.data = w.data-lr*w.grad.data          #.data  gives us access to w
                                            # w updated by .1 by iteration and there for y_hat
    w.grad.data.zero_()                     # solves the derivative using data a a certain index
                                            #see also optim.SGD
############################################################################################################################################################
############################################################################################################################################################





####################################################################### GRADIENT DESCENT WITH 2 TERMS ####################################################
############################################################################################################################################################
# If we add another term b this also must be 
# altered incrementaly to find the best fit.....

 # set up a core linear function
################################
def forward(x):
    y=w*x+b                                  # a way to generate y_hat w will change by .1 in our fuction
    return y

#itialize the weights bias, noise x and y values
################################
w = torch.tensor(-15,requires_grad=True)
b = torch.tensor(-10,requires_grad=True)
x = torch.tensor.arange(-3,3,0.1.view(-1,1)) # range from -3 to 3 by increments of .1 view adds another dimension not sure why this is needed
noise =  1 * x - 1
y = noise +  .1 * torch.randn(x.size())


 # set up loss/cost function
################################
def criterion(yhat,y):                       # a loss fuction that compares y valaues
    return torch.mean((yhat-y)**2)


# Loop it all together
################################
for epoch in range(15):                     # for every element of data multiply 10
    Yhat = forward(x)                       # but why do it 4 times?????
    loss = criterion(Yhat,y)
    loss.backward()                         # gives us the partial derivative of 
    #w.grad                                 # returns derivative at -10
    w.data = w.data-lr*w.grad.data          # .data  gives us access to w. it is updated by .1 (and thus y_hat)
    b.data = b.data-lr*b.grad.data          # .data  gives us access to b it is updated by .1 (and thus y_hat)
    w.grad.data.zero_()                     # solves the partial derivative using data a a certain index
    b.grad.data.zero_()                     # solves the partial derivative using data a a certain index
                                            # see also optim.SGD
############################################################################################################################################################
############################################################################################################################################################







############################################################### STOCHASTIC GRADIENT DESCENT WITH 2 TERMS ####################################################
############################################################################################################################################################
#itialize the weights bias, noise x and y values
###############################################
w = torch.tensor(-15,requires_grad=True)
b = torch.tensor(-10,requires_grad=True)
x = torch.tensor.arange(-3,3,0.1.view(-1,1))      # range from -3 to 3 by increments of .1 view adds another dimension not sure why this is needed
f = 3*x 
y = f+.1*torch.randn(x.size())                    # add random noise and get y values

import matplotlib.pyplot as plt                   # plt.plot(x,y)
plt.plot( x.numpy,y.numpy())                      # numpy changes tensors to arrays for plotting          



# set up the model, loss/cost and learning rate
####################################
def forward(x):
    return w*x + b                            # a way to generate y_hat w will change by .1 in our fuction this time the fuction uses one point at a time???

def criterion(yhat,y):                        # a loss fuction that compares y valaues
    return torch.mean((yhat-y)**2)



lr = .1

LOSS = []
# Loop it all together
#######################
for epoch in range(4):                          # for every element of data multiply 10
    for x,y in zip(x,y):                         # STOCHASTIC GRADIENT DESCENT calculates loss for each point as a SCALAR!!!!!!!!!!!!!!!!!!
        Yhat = forward(x)                       # but why do it 4 times?????
        loss = criterion(Yhat,y)
        loss.backward()                         #gives us the derivative of 
        #w.grad                                 #returns derivative at -10
        w.data = w.data-lr*w.grad.data          #.data  gives us access to w it is updated by .1 (and thus y_hat)
        b.data = b.data-lr*b.grad.data          #.data  gives us access to b it is updated by .1 (and thus y_hat)
        w.grad.data.zero_()                     # solves the partial derivative using data at a certain index
        b.grad.data.zero_()                     # solves the partial derivative using data at a certain index
        LOSS.append(loss.item())                # spits out the loss value to a list we can inspect later
                                                # see also optim.SGD
############################################################################################################################################################
############################################################################################################################################################






################################################################# optim.SGD ################################################################################
############################################################################################################################################################

from torch.nn import Linear
model = Linear(1,1)                              #makes our Linear model, in_features is the number of columns out_features is the size of each sample

from torch import nn, optim
optimizer = optim.SGD(model.parameters(), lr = .1)   
optimizer.step()                                # solveS ALL partial derivatives on iteration step no need for lines 160 thru 163.....
                     
                     
#  this shortens the code if you have alot of parameters/partial derivatives to find      

optimizer = optim.SGD(forward.parameters(), lr = .1) 
for epoch in range(4):                          # for every element of data multiply 10
    for x,y in zip(x,y):                         # STOCHASTIC GRADIENT DESCENT calculates loss for each point as a SCALAR!!!!!!!!!!!!!!!!!!
        Yhat = forward(x)                       # but why do it 4 times?????
        loss = criterion(Yhat,y)
        loss.backward()                         # This solves the  derivative 
        optimizer.step()                        # updates the optimizer
##########################################################################################################################################









######################################################### OBJECT ORIENTED APPROACH ########################################################################
############################################################################################################################################################
#TYPICALLY WE SPLIT THE DATA INTO THREE PARTS
#TRAINING DATA
#TEST DATA
#VALIDATION DATA
# The Training data is used to make our various models using gradient descent
# each will have its on micro parameters like BATCHSIZE and LEARNING RATE (lr)
# and then ALL MODELS are tested against the Validation data  for fittment in the form of another loss/cost function
# the model with the lowest cost on the Validation data is the winnner
       
citerion = nn.MSELoss()
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Here we express the abive code as a
# class where the user can enter variables w and b
# x is set but can also be made a variable of the class
class data_set(Dataset):
    def __init__(self,w,b, train = True):
        #super(data_set.self).__init__()                   #inherit parent attributes 
        self.w = torch.tensor(w,requires_grad=True)
        self.b = torch.tensor(b,requires_grad=True)
        self.x = torch.tensor.arange(-3,3,0.1.view(-1,1)) # range from -3 to 3 by increments of .1 view adds another dimension not sure why this is needed
        self.f = w*x+b
        self.y= f + .1*torch.randn(x.size()) 
        self.len = self.x.shape[0] 
        if train == True:
            self.y[0] = 0                                  #value have been set (ideally we'd sample)
            self.y[50:55] = 20                             #values have been set
        else:
            pass

    def __getitem__(self,index):
        return self.x[index],self.y[index]
     
    def __len__(self):
        return self.len

# with this new class we can make objects the contain the equation, extra features,
# and fresh data
dataset = data_set(w=-15,b=-10, train=True) # True changes the data
valdata = data_set(w=-15,b=-10,train=False) # False leaves the data as is



## we can also create a new object for linear regression
# that inherits much of the nn.Module functions from the torch.nn library
import torch.nn as nn
class LR(nn.Module):                                          # name of our object
    def __init__(self, input_size, output_size): 
        super(LR.self).__init__()                             # inherit parent attributes 
        self.linear = nn.Linear(input_size, output_size)      # thanks to inheritance we can call the "Linear" f(x) using variables

    def forward(self,x):                                      # x allows us to reference the variable in the function
        out = self.linear(x)
        return out
    

# with this new class we can make new objects with other features not included with Linear
   # DataLoader takes a subset of our data acccording to batch

epochs = 10
learning_rates = [.00001,.0001,.001,.01,.1,1]

trainloader = DataLoader(dataset = dataset, batch_size=1)        
model = LR(input_size=2,output_size=1)                           # model is now a linear regression of the appriate  features and sample rows
optimizer = optim.SGD(model.parameters(), lr = learning_rates)
validation_error = torch.zeros(len(learning_rates))
models = []


# what we previously made a function for loss  torch.nn has MSELoss()
criterion = nn.MSELoss()





from torch import optim
for i,learning_rate in enumerate(learning_rates):
    model= LR(input_size = 1, output_size = 1)                     # model is now a linear regression of the appropriate  features and sample rows
    optimizer = optim.SGD(model.parameters(), lr = learning_rate )

    for epoch in range(epochs):
        # this generates 60 models -3 -3 by .1
        for x,y in trainloader:       # for every point in the sampled data of batch size 1
            yhat = model(x)           # predict a y value 
            loss = criterion(yhat,y)  # calculate a loss for that point
            optimizer.zero_grad()     # resets the gradient
            loss.backward()           # solves each/all derivative  at that point which stores the best criterion
            optimizer.step()          # updates the optimizer

    # for each epoch there are 60 models
    yhat2 = model(dataset.x)          # for every set of x the model gives 1 output as defined earlier!   
    loss = criterion(yhat,dataset.y)
    validation_error[i]= loss.item()
    models.append(model)

    # In each epoch the best model is compared to the validation data
    # the model and its cost are appended to a list
    yhat3 = model(valdata.x)         
    loss = criterion(yhat,valdata.y)
    validation_error[i]= loss.item()
    models.append(model)

# In the end there should be one model per learning rate  
     






######################################################### EARLY STOPPING ########################################################################
############################################################################################################################################################
# In early stopping instead of collecting all models  and their costs on validation data ie min (Model 1 | Model 2 | Model 3 | model 4)
# we do this after each iteration of gradient descenet training                          ie if: Model 1 <= Model 2 <= Model 3    if model 4 !<= model 3 stop!!!!!
# The Training data is used to make our various models using gradient descent
# and then EACH MODEL AT EACH ITERATION is tested against the Validation datato for fittment in the form of a COST
# if the latest model is better than the cutoff we make this our new cutoff and save that model to a file     
# using out data class and model class object from above we set a cutoff of min_loss that gets updated
criterion = nn.MSELoss()

epochs = 10
learning_rates = [.00001,.0001,.001,.01,.1,1]
validation_error = torch.zeros(len(learning_rates))
models = []
min_loss = 100 #our early stop value for validation loss/cost


checpoint_path = 'checkpoint_model.pt'              # sometimes we need to write out each epoch
checkpoint = {'epoch':None,                         # assign each epoch here 
              'model_state_dict':None,
              'optimizer_state_dict':None,
              'Loss':None} 


# Training with EARLY STOPPING with checkpoints
from torch import optim
for i,learning_rate in enumerate(learning_rates):
    model= LR(input_size=1,output_size=1)            # model is now a linear regression of the appriate  features and sample rows
    optimizer = optim.SGD(model.parameters(), lr = learning_rate )

    for epoch in range(epochs): 
        # 60 models are generated -3 to 3 by .1
        for x,y in trainloader:       # for every point in the sampled data of batch size 1
            yhat = model(x)           # predict a y value 
            loss = criterion(yhat,y)  # calculate a loss for that point
            optimizer.zero_grad()     # resets the gradient
            loss.backward()           # solves each/all derivative  at that point
            optimizer.step()          # updates the optimizer
            
            # if the loss is greater we go to the next model of  the 60 one is returned
            loss_train = criterion(model(trainloader.x),trainloader.y).item()
            loss_val= criterion(model(valdata.x),valdata.y).item()
            if loss_val < min_loss:  
                value=epoch
                min_loss = loss_val
                torch.save(model.state_dict(),'filename_best_model.pt')

    yhat2 = model(dataset.x)   # for every set of x the model gives 1 output as defined earlier!       
    loss = criterion(yhat,dataset.y)
    validation_error[i]= loss.item()
    models.append(model)

    # In each epoch the best model is compared to the validation data
    # the model and its cost are appended to a list
    yhat3 = model(valdata.x)         
    loss = criterion(yhat,valdata.y)
    validation_error[i]= loss.item()
    models.append(model)

    checkpoint['epoch']= epoch
    checkpoint['model_state_dict']= model.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']= loss
    checkpoint['epoch']= epoch
    torch.save(checkpoint,checpoint_path)


# we load the state dictionary and the loss and resume if need start and stop at a certain iteration
optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['Loss']


# in the end there should be 1 model per learning rate soa  list of 6

