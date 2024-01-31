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
        out = self.linear(x)                            # a new function that takes the linear model of x and puts it thru the Sigmoid function
        return torch.sigmoid(out)
    
    def __call__(self,x):                       # I think by using __call__ you dont need to name the function
        out = self.linear(x)
        return torch.sigmoid(out)

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
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len
    
   # def __getitem__(self, idx):
    #    if torch.is_tensor(idx):
     #       idx = idx.tolist()

    
 

##########################################################################################################################################

dataset = data_set(df)
# I can't get this into the train loader
# DataLoader needs the dataset class to have certain attributes
# in order to make a useful train loader object 
trainloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)






# make validation and training data 9879/2 and then the rest is validation
# it will split df but now it will be a tensor object

# train_data, val_data = torch.utils.data.random_split(df,[4939,(9879-4939)], generator=torch.Generator().manual_seed(1))

# now the df is an object of a custom built class with the data divided as x an y tensors
# make training and validation data objects and feed into the DataLoader()

#train_data  =  data_set(train_data)  # :(  
#val_data  =  data_set(val_data)     # :( 

# trainloader = DataLoader(dataset=train_data, shuffle=True, batch_size=1) :(
# val_loader = DataLoader(dataset=val_data, shuffle=True, batch_size=1)    :(   
 







###########################################################TRAIN THE MODEL with Gradient Descent ############################################

criterion = nn.BCELoss()                                           # 3d data with 2 samples
#trainloader = DataLoader(dataset = train_data ,batch_size=1)           # get training data
model = logistic_regression(39)                                # Feed our model object based on data dimension!!!!
optimizer = optim.SGD(model.parameters(),lr=0.01)                  # Stochastic gradient descent optimizer
criterion = nn.BCELoss()
checpoint_path = 'checkpoint_model.pt'              # sometimes we need to write out each epoch
checkpoint = {'epoch':None,                         # assign each epoch here 
              'model_state_dict':None,
              'optimizer_state_dict':None,
              'Loss':None} 




for epoch in range(100):
    for x,y in trainloader:                    # for every iterations of x y in our new data class
        yhat = model(x)                        # predict a y value from all features except blue wins???
        loss = criterion(yhat,y)               # calculate a CROSS ENTROPY LOSS for those points vs our predictor
        optimizer.zero_grad()                  # resets the gradient
        loss.backward()                        # creates a set of derivatives from the loss equation and solves 
        optimizer.step()                       # update the gradient descent optimizer
    checkpoint['epoch']= epoch
    checkpoint['model_state_dict']= model.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['Loss']= loss
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
trainloader.dataset 

train_loader_iter = iter(trainloader)
# I can see the object type
type(train_loader_iter) 
# creating an object with the proper attributes has allowed me
# to iterate thru the trainloader output from DataLoader !!! :) 
for i in enumerate(train_loader_iter):
      print(i)

# trainloader as a DataLoader object will have the following attributes

['_DataLoader__initialized','__annotations__','__class__','__class_getitem__',
 '__delattr__','__dict__','__dir__','__doc__','__eq__','__format__','__ge__',
 '__getattribute__', '__getstate__','__gt__','__hash__','__init__',
 '__init_subclass__','__iter__','__le__','__len__','__lt__','__module__',
 '__ne__','__new__','__orig_bases__','__parameters__','__reduce__','__reduce_ex__',
 '__repr__','__setattr__','__sizeof__','__slots__','__str__','__subclasshook__',
 '__weakref__','_auto_collation','_get_iterator','_index_sampler','_is_protocol',
'check_worker_number_rationality','multiprocessing_context']


# KEY THINGS TO CONSIDER

# making the Data an object allows you to define which columns are x and which are y
# the the model needs to know how many features to expect
# the df converted to a tensor as x features shold reflect this 
# the Sigmoid function on the linear model is what makes it logisitic
# the Dataloader needs certain attributes to work...consider this when making objects
# dir() allows you to examine attributes














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
