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

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# or if working with tensors

import torch
import matplotlib  as plt
z = torch.arange(-100,100,.1).view(-1,1)
y_hat = torch.sigmoid(z)
plt.plot((z.numpy),y_hat)



# another way is with nn module
# first we make a linear model
# pass the data in to make a prediction
# pass that prediction to the sigmoid function
# to plot it from 0 to 1

import torch.nn as nn

nn.Sequential #chains models together
nn.Linear     # A module that has a linear formula
nn.Sigmoid    # A module that has a Logistic regression formula
model = nn.Sequential(nn.Linear(1,1),nn.Sigmoid)
y_hat = model(z)



######################################################### OBJECT ORIENTED APPROACH ########################################################################
############################################################################################################################################################

# just as we subclassed the Linear module we can do the 
# same and make our own class based on the Linear regression module

class logistic_regression(nn.Module):                                # our class with the objects we;d like to copy
    def __init__(self,in_size):                                      # initalize self along with any other arguments
        super(logistic_regression,self).__init__()                   # give logistic regression the features of nn.Module
        self.linear = nn.Linear(input_size= in_size,output_size=1)   # new variable linear is a model based on Linear()
    
    def forward(self,x):
        x = torch.sigmoid(self.linear(x))                            # a new function that takes the linear model of x and puts it thru the Sigmoid function



# setup virtual environment and install at command line 
# python -m venv my_env
# pip3 install sklearn 
# or
import sklearn
from sklearn.linear_model import LogisticRegression1