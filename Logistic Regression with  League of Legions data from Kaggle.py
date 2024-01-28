import torch
import pandas as pd
from torch import optim
from torch.utils.data.dataloader import DataLoader



######################################################### READ IN DATA ##############################################################
df = pd.read_csv("C:/Users/jabba/OneDrive/Desktop/Sandbox/Python/high_diamond_ranked_10min.csv")
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
# tensor_df = torch.Tensor(np.array(df))



# make validation and training data

train_data, val_data = torch.utils.data.random_split(df,[4939,(9879-4939)], generator=torch.Generator().manual_seed(1))
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=1)
val_loader = DataLoader(dataset=val_data, shuffle=False, batch_size=1)
##########################################################################################################################################








############################################### Create a Logistic Regression Model Object#################################################

import torch.nn as nn
class logistic_regression(nn.Module):                                # our class with the objects we;d like to copy
    def __init__(self,in_size):                                      # initalize self along with any other arguments
        super(logistic_regression,self).__init__()                   # give logistic regression the features of nn.Module
        self.linear = nn.Linear(in_features = in_size , out_features=1)   # new variable linear is a model based on Linear()
    
    def forward(self,x):
        out = nn.sigmoid(self.linear(x))                            # a new function that takes the linear model of x and puts it thru the Sigmoid function
        return out

##########################################################################################################################################









###########################################################TRAIN THE MODEL with Gradient Descent ############################################

criterion = nn.BCELoss()                                           # 3d data with 2 samples
trainloader = DataLoader(dataset = train_data ,batch_size=1)           # get training data
model = logistic_regression(40)                                # Feed our model object based on data dimension!!!!
optimizer = optim.SGD(model.parameters(),lr=0.01)                  # Stochastic gradient descent optimizer
criterion = nn.BCELoss()
checpoint_path = 'checkpoint_model.pt'              # sometimes we need to write out each epoch
checkpoint = {'epoch':None,                         # assign each epoch here 
              'model_state_dict':None,
              'optimizer_state_dict':None,
              'Loss':None} 




for epoch in range(100):
    for my_list,blueWins in trainloader:       # for every point in the sampled data of batch size 1
        yhat = model(my_list)                  # predict a y value from all features except blue wins???
        loss = criterion(yhat,y)               # calculate a CROSS ENTROPY LOSS for those points vs our predictor
        optimizer.zero_grad()                  # resets the gradient
        loss.backward()                        # creates a set of derivatives from the loss equation and solves 
        optimizer.step()                       # update the gradient descent optimizer
    checkpoint['epoch']= epoch
    checkpoint['model_state_dict']= model.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']= loss
    checkpoint['epoch']= epoch
    torch.save(checkpoint,checpoint_path)
##########################################################################################################################################

# at the end there should be a model for each step or sample 
# each must be compared to validation data to see which has the lowest
# CROSS ENTROPY LOSS (basically the loss for classification)
# to increase the speed consider chagning the batch size in the Data loader








#prints the train data:)
train_data.dataset 

### "for every my_list,blueWins in train_loader" (x1,x2,x3.....x40,y) isn't working
### Once I see it maybe I can figure out how to iterate thru it.
train_loader.dataset 

train_loader_iter = iter(train_loader)
# I can see the object type
type(train_loader_iter) 
### Trying to look at the train_loader data but can't :(
for i in enumerate(train_loader_iter):
      print(i)



















# lets make it a DATA OBJECT for fun

class my_data():
    def __init__(self,path_to_csv):
        self.path_to_csv = path_to_csv
        self.df = pd.read_csv(self.path_to_csv)

    def print(self):
        print(self.df)
 
    def colnames(self):
        print(self.df.columns)

    def wins(self):
        return self.df['blueWins']
