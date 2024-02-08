import numpy as np
import torch
import pandas as pd
from torch import optim
from torch.nn.modules.activation import Sigmoid
from torch.utils.data.dataloader import DataLoader


import os


Path = os.path.join(os.getcwd(), "high_diamond_ranked_10min.csv")

######################################################### READ IN DATA ##############################################################
df = pd.read_csv(Path)
print(df.columns)                      # blue wins is our classifier
print(df.shape)                        # 40 features with 9879 samples
my_list1 = df.columns.tolist()         # list of features
my_list = my_list1.remove('blueWins')               # list except the our y predictor

# we could split the data frame into x and y 
# but its better to partion the data into Train and Validation data
#x = df.drop('blueWins', axis=1)
#y = df['blueWins']

# convert to a tensor 
# import torch
# import numpy as np
# tensor_df = torch.Tensor(df.to_numpy())





############################################### Create a Logistic Regression Model Object#################################################

import torch.nn as nn
class logistic_regression(nn.Module):                                # our class with the objects we;d like to copy
    def __init__(self,in_size):                                      # initalize self along with any other arguments
        super(logistic_regression,self).__init__()                   # give logistic regression the features of nn.Module
        self.linear = nn.Linear(in_features = in_size , out_features =1 )   # new variable linear is a model based on Linear()
    
    def forward(self,x):
        out = self.linear(x)                    # a new function that takes the linear model of x and puts it thru the Sigmoid function
        return torch.sigmoid(out)               # the sigmoid makes it logistical and non binary
    
    def __call__(self,x):                       # I think by using __call__ you dont need to name the function
        out = self.linear(x)
        return torch.sigmoid(out)               # the sigmoid makes it logistical and non binary

##########################################################################################################################################

 

############################################### Create a Dataset Class #################################################
# I think once ew make train data and val data an object
# we can reference x and y in our training loop
# if we first define them as tensors  x and y
from torch.utils.data import Dataset

class data_set(Dataset):
    def __init__(self,df):
        super(data_set, self).__init__()                                      
        self.x = torch.Tensor(df.iloc[:, np.r_[0, 2:40]].to_numpy()) # matrix 1  should match model input
        self.y = torch.Tensor(df.iloc[:,[1]] .to_numpy())             # matix 2 
        self.len = self.x.shape[0] 

    def __getitem__(self,index):
        return self.x[index],self.y[index]                          # important attribute to have

    def __len__(self):                                              # important attribute to have
        return self.len
##########################################################################################################################################

dataset = data_set(df)      # converts df to a tensor object
dataset[0]                  # will look at first sample


  # DataLoader needs the dataset class to have certain attributes
  # in order to make a useful train loader object :(   )
# trainloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)

# train_data, val_data = torch.utils.data.random_split(df,[4939,(9879-4939)], generator=torch.Generator().manual_seed(1))

# train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=1) 
# val_loader = DataLoader(dataset=val_data, shuffle=True, batch_size=1)      
 






import torch.utils.data as data
train_data_tensor, val_data_tensor = data.random_split(dataset,[4939,(9879-4939)],generator=torch.Generator().manual_seed(1))


# now the df is an object of a custom built class with the data divided as x an y tensors
# make training and validation data objects and feed into the DataLoader()


train_loader = DataLoader(dataset=train_data_tensor, shuffle=True, batch_size=1) 
val_loader = DataLoader(dataset=val_data_tensor, shuffle=True, batch_size=1)      
 







###########################################################TRAIN THE MODEL with Gradient Descent ############################################

criterion = nn.BCELoss()                                           # 3d data with 2 samples
#trainloader = DataLoader(dataset = train_data ,batch_size=1)           # get training data
model = logistic_regression(39)                                # Feed our model object based on data dimension!!!!
#learning_rates = [.00001,.0001,.001,.01,.1,1]
optimizer = optim.SGD(model.parameters(),lr= .01)   # Stochastic gradient descent optimizer for each learning rate
criterion = nn.BCELoss()
checpoint_path = 'checkpoint_model.pt'              # sometimes we need to write out each epoch
checkpoint = {'epoch':None,                         # assign each epoch here 
              'model_state_dict':None,
              'optimizer_state_dict':None,
              'Loss':None} 
models = []
loss_list = []




for epoch in range(10):
    for x,y in train_loader:                   # for every iterations of x y in our new data class
        yhat = model(x)                        # predict a y value from all features/predictors
        loss = criterion(yhat,y)               # calculate a CROSS ENTROPY LOSS for that point vs our predictor
        optimizer.zero_grad()                  # resets the gradient
        loss.backward()                        # creates a set of derivatives from the loss equation and solves 
        optimizer.step()                       # update the gradient descent optimizer with a model for each learning rate
    #I think this updates and returns a single model
    # the as the optimizer goes thru each point it updates 
    # the best fit for each optimizer returning a model for each learning rate
    # For each epoch I think I'm getting back 5 models one for every Learning rate
    checkpoint['epoch']= epoch
    checkpoint['model_state_dict']= model.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['Loss']= loss
    #print(epoch)
    torch.save(checkpoint,checpoint_path)


     
# ????  is the model considered optimized now 
# ????  we shold be getting loss on Training and validation data
# ????  but why for every epoch isnt it just the same set of models 
   
yhat2 = model(train_data_tensor.dataset.x)          
loss = criterion(yhat2,train_data_tensor.dataset.y)
#validation_error[i]= loss.item()
models.append(model)
loss_list.append(loss)
##########################################################################################################################################

# at the end there should be a model for each step or sample 
# each must be compared to validation data to see which has the lowest
# CROSS ENTROPY LOSS (basically the loss for classification)
# to increase the speed consider chagning the batch size in the Data loader








# KEY THINGS TO CONSIDER

# making the Data an object allows you to define which columns are x and which are y
# the the model needs to know how many features to expect
# the df converted to a tensor as x features shold reflect this 
# the Sigmoid function on the linear model is what makes it logisitic
# the Dataloader needs certain attributes to work...consider this when making objects
# dir() allows you to examine attributes












