
import torch
# Just like in Caret from our Data Science  and Machine learning course
# PYtorch offers the ability to make prediction on linear and non linear equations
# for example the formula  y = mx + b can be derived from training data
# we then make predicitions on it .
# here we user tensor data as x and y setting grad to true alerts
# torch that the tensor will be used for a prediction

w = torch.tensor(2,requires_grad=True)
b = torch.tensor(-1,requires_grad = True)
x = torch.tensor([1]) # scalar of 1

y = w*x+b
#or
def my_function(x):
    y=w*x+b
    return y

y_hat=my_function(x)

# for 1d tensor 
# |1|
# |2|
# |3|
# we can apply the same linear function to every row
# in the tensor
x = torch.tensor([1,2,3])




# Pytorch can create some of the shelf models
# such as linear but we create our own if need be

from torch.nn import Linear
torch.manual_seed(1)

model = Linear(in_features=1,out_features=1) # in_features is the number of columns out_features is the size of each sample
x = torch.tensor([1])                        # 1d scalar 
x = torch.tensor([1,2])                      # a 1d tensor applies each value of x to the model

                                             #or                                            
x = torch.tensor([[1],[2]])                  # also a 1d tensor applies each value of x to the model??? seems like 2d to me but whatever
y_hat=model(x)                               # make predictions

# for Higher Dimensions
model = Linear(in_features=2,out_features=2)
[x,z] = torch.tensor([[1,2],[7,4]])          # a 2d tensor in_features would be 2 out_features would also be 2
                                             #or
x = torch.tensor([1,2])
z = torch.tensor([7,4])
y_hat=model(x,z)                            
list(model.parameters())                     # returns the slope and  bias for the model in tensor form




# We can clone the linear model Module and customize it as a new class

import torch.nn as nn

class my_model(nn.Module):
    def __init__(self,input_size,output_size): #initalize self and attributes
        super(my_model,self).__init__()        # ensures attributes of all nn.Modules are inherited
                                               # this allows us to reference nn.Linear
        self.linear=nn.Linear(input_size,      # takes the input and output attrubutes applies them to 
                              output_size)     # the Linear module and assigns a new attribute to self
    
    def forward(self,x):
        out=self.linear(x)
        return out
    
    def __call__(self,x):                       # I think by using __call__ you dont need to name the function
        out=self.linear(x)
        return out
    
# new class can be used with the added functions within
# its almost like cloning and environment
my_model.forward(x)
# thanks to def __call__
my_model(x)

model.state_dict() #returns a dictionary  