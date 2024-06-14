# Working with Multiple Linear regression
# is simple fitting a Linear model in 3 or more Dimensions
# y_hat = x1*w1 + x2*w2 + x3*w3 + b


# consider the data frame....
# we have 3 dimensions or features
# and 4 rows of data or samples


# For the data
# pytorch would represent the DIMENSIONS in the following way
# torch.tensor[[x1,x2,x3],[x1,x2,x3],[x1,x2,x3],[x1,x2,x3].....] or
# torch.tensor[[x1,x2,x3],[x1,x2,x3],[x1,x2,x3],[x1,x2,x3].....]



# when we make our new model object class that accounts for this formula 
# y_hat = x1*w1 + x2*w2 + x3*w3 + b
# nn.Linear(input_no,output_no)
# the number of features (input_no) would be 3 and 
# the number of samples (output_no) would be 4, a y_hat for each sample 

# pytorch would represent the WEIGHTS with 
# torch.tensor[w1,w2,w3]
# looking back at our table pytorch can multiply this matrix times a 
# tensor matching the  dimension no
# see rules of matrix math / dot product

# partial derivatives of the loss function and gradient descent  can find which 
# model yields the smallest cost  
# the partial derivative equations are arranged as a vector |d1,d2,d3|
# fortunately nn.MSEloss() does this

# in any case solving this system  |d1,d2,d3| gives us the direction we need to move 
# to get the biggest reduction in loss. At this point the 
# "gradient vector |d1,d2,d3| and unit vector |x1,x2,x3| are parralell 
# this happens when the variable "cosine theta" is set to zero"


# im a bit confused at this step?
# is the data used iterively or does its just cycle thru random numbers?
# or both? maybe the constant for the partial derivative comes from the data?

# I think the optimizer solves the derivative by returning a vector of lowest cost
# for each data point. each represents a direction

# in the epoch loop we go thru each to see which fits the validation the best
# 

# most of the code is the same except our linear model will have a higer input_no. 
# the output only reflects the sample number.
# the table below has 4 features and 2 samples
#     x1 ,  x2,  x3,   x4
#      1    3.3  .3    7
#      4    5.7   1    0

# I think we only have to change x tensor and the LR model 



######################################################### OBJECT ORIENTED APPROACH FOR MULTI DIMENSIONAL DATA ###################################################
##################################################################################################################################################################
 
# The code is similar for 2d data except the tensors we use
# This is the beauty of pytorch the many dimenions of x are fed in as a single tensor
# dot product math iterates over everything so we can be less explicit
# with our formula simple linear formula of  w * x + b

# our outputs simple come  out in tensor list format
        
citerion = nn.MSELoss()
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# NOTICE HOW THE TENSORS HAVE MULTIDIMENSIONS NOW
class data_set(Dataset):
    def __init__(self,w,b, train = True):
        #super(data_set, self).__init__()                   #inherit parent attributes 
        self.w = torch.tensor(w,requires_grad=True)                                  # this will be 4d ([w1,w2,w3,w4]
        self.b = torch.tensor(b,requires_grad=True)                                  # I think bias should be 4d as  well ([b1,b2,b3,b4])
        self.x = torch.tensor.arange( [ [1, 3.3 , .3 , 7] , [ 1 ,3.3,  .3 ,   7]  ]) # the data  is a 4d tensor with 2 samples
        self.f = w * self.x + b
        self.y= self.f + .1*torch.randn(x.size()) 
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
model = LR(input_size=4,output_size=1)                           # model is now a linear regression of the appriate  features and sample rows
optimizer = optim.SGD(model.parameters(), lr = learning_rates)
validation_error = torch.zeros(len(learning_rates))
models = []


# what we previously made a function for loss  torch.nn has MSELoss()
criterion = nn.MSELoss()





from torch import optim
for i,learning_rate in enumerate(learning_rates):
    model= LR(input_size = 4, output_size = 2)                     # model is now a linear regression of the appropriate  features and sample rows
    optimizer = optim.SGD(model.parameters(), lr = learning_rate )

    for epoch in range(epochs):
        # this generates 60 models -3 -3 by .1
        for x,y in trainloader:       # for every point in the sampled data of batch size 1
            yhat = model(x)           # predict a y value 
            loss = criterion(yhat,y)  # calculate a loss for that point
            optimizer.zero_grad()     # resets the gradient
            loss.backward()           # creates a set of derivatives 
            optimizer.step()          # solves each/all derivative  at that point which stores the best criterion

    # for each epoch there are as many models as there are samples. 
    # each needs to produce a prediction or  y_hat from the data
    # loss is calculated against the actual y data
    # the least of which is the winner
    yhat2 = model(dataset.x)          # for every set of x the model gives 1 output as defined earlier!   
    loss = criterion(yhat,dataset.y)
    validation_error[i]= loss.item()
    models.append(model)

    # In each epoch the best model is compared to the validation data
    # the model and its cost are appended to a list as well 
    yhat3 = model(valdata.x)         
    loss = criterion(yhat,valdata.y)
    validation_error[i]= loss.item()
    models.append(model)

# In the end there should be one model per learning rate  
