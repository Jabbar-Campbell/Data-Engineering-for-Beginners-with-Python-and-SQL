#                                                                                    LOGISTIC REGRESSION
#In Linear regression we fit a line to the data using gradient descent for each point
# LOGISTIC regression is when data belongs to 1 or more classes and we need to predict 
# to which they belong

# A threshold function returning 0 or 1 is useful. with it we can test it on our data 
# and obtain probablities. (80 percent of the data is a 1 20 percent 0)
# this can be compared to the breakdown of the actual data

# a logistic regression function is simlar but instead of a hard 0 or 1 
# it returns a probablity between 0 an 100 due to its 
# sigmoid shape

import dataclasses
import math
from torch import optim
from torch.utils.data import dataloader

from torch.utils.data.dataloader import DataLoader
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# or if working with tensors

import torch
import matplotlib  as plt
z = torch.arange(-100,100,.1).view(-1,1)
y_hat = torch.sigmoid(z)
plt.plot((z.numpy),y_hat)




# another wayto create a LOGISTIC REGRESSION MODEL is with nn module
# first we make a linear model
# pass the data in to make a prediction
# pass that prediction to the sigmoid function
# to plot it from 0 to 1

import torch.nn as nn

nn.Sequential                                              # chains models together
nn.Linear                                                  # A module that has a linear formula
nn.Sigmoid                                                 # A module that has a Logistic regression formula
model = nn.Sequential(nn.Linear(input_size=1,
                                output_size=1),nn.Sigmoid) # input size reflect dimensions of the data
y_hat = model(z)

# Like linear regression logistic regression has a Loss or cost function 
# itS call CROSS ENTROPY lOSS  and is expressed as follows
# similar to the confusion matrix in caret it looks at what was correctly classfied
# and what wasnt. The derivative and Gradient descent that minimizes this loss
# is our true best model for Logistic Regression
def citerion(yhat,y):
    CROSS_ENTROPY_lOSS  =  1*torch.mean(y*torch.log(yhat)+1-y*torch.log(1-yhat))
    return CROSS_ENTROPY_lOSS
#or
import sklearn 
sklearn.metrics.log_loss
# or

criterion = nn.BCELoss

######################################################### OBJECT ORIENTED APPROACH ########################################################################
############################################################################################################################################################

# just as we subclassed the Linear module we can do the 
# same and make our own class based on the Linear regression module
# x can be a 1d,2d, or 3d tensor
# torch.tensor[1]           scalar
# torch.tensor[[1],[100]]                                            2 samples 1 d input size is 1
# torch.tensor[[1,4,.8],[100,450,75]]                                2 samples 3 d input size is 3

import torch.nn as nn
class logistic_regression(nn.Module):                                # our class with the objects we;d like to copy
    def __init__(self,in_size):                                      # initalize self along with any other arguments
        super(logistic_regression,self).__init__()                   # give logistic regression the features of nn.Module
        self.linear = nn.Linear(input_size= in_size,output_size=1)   # new variable linear is a model based on Linear()
    
    def forward(self,x):
        out = nn.sigmoid(self.linear(x))                            # a new function that takes the linear model of x and puts it thru the Sigmoid function
        return out



trainloader = DataLoader(dataset = dataset ,batch_size=1)           # get training data
model = logistic_regression(1,1)                                    # establish model based on data
optimizer = optim.SGD(model.parameters(lr = 0.01))                  # Stochastic gradient descent optimizer

for epoch in range(epochs):
        for x,y in trainloader:       # for every point in the sampled data of batch size 1
            yhat = model(x)           # predict a y value 
            loss = criterion(yhat,y)  # calculate a CROSS ENTROPY LOSS for that point
            optimizer.zero_grad()     # resets the gradient
            loss.backward()           # creates a set of derivatives and solves 
            optimizer.step()          # update the gradient descent optimizer

#
# setup virtual environment and install at command line 
# python -m venv my_env
# pip3 install sklearn 
# or
import sklearn
from sklearn.linear_model import LogisticRegression1