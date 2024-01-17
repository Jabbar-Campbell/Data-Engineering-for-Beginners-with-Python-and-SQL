###                                                  GRADIENT DESCENT
# given a set of points there will several lines that can
# fit the data. The best line is represented as such where
# the loss fucntion (y_hat-y)^2 of all the points aka 
# Root mean square error or RMSE is the smallest  
# RMSE = √[ Σ(y_hat – y)^2 / n ]
# or

# pip3 install Scikit-learn at command line
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_actual, y_predicted))

# to find this we need to take the Derivative of our RMSE 
# RMSE.backward()     ---see also .backward()----
# my_parameters.grad  ---see also .grad----
# and swap in values associated with slope and intercept. The  values that give us the
# lowest RMSE is the best fit.
# in linear regression fittment and there fore predictions are dependent on 2 things
# the slope changeing and the intercept chaneging

# this process of swapping in values into the Derivative of our loss f(x) to get new batches of predictions 
# summarized by the RMSE calculation is called GRADIENT DESCENT








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



# set up loss/cost and learning rate
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
    for x,y in zip(XY):                         # STOCHASTIC GRADIENT DESCENT calculates loss for each point as a SCALAR!!!!!!!!!!!!!!!!!!
        Yhat = forward(x)                       # but why do it 4 times?????
        loss = criterion(Yhat,y)
        loss.backward()                         #gives us the derivative of 
        #w.grad                                 #returns derivative at -10
        w.data = w.data-lr*w.grad.data          #.data  gives us access to w it is updated by .1 (and thus y_hat)
        b.data = b.data-lr*b.grad.data          #.data  gives us access to b it is updated by .1 (and thus y_hat)
        w.grad.data.zero_()                     # solves the partial derivative using data at a certain index
        b.grad.data.zero_()                     # solves the partial derivative using data at a certain index
        LOSS.append(loss.item())                # spits out the loss value to a list we can inspect later
############################################################################################################################################################
############################################################################################################################################################



# DATA LOADERS use the above code in a class object
# why i have no idea.....