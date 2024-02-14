from matplotlib import pyplot as plt
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
    # might not need
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
        self.y = torch.Tensor(df.iloc[:,[1]] .to_numpy())  # required true is necessary for gradients calculation            # matix 2 
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
trainloader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
   # dataiter = iter(trainloader)
   # data = dataiter.next()
   # features , labels = data
   # print(features , labels)
   # train_data, val_data = torch.utils.data.random_split(df,[4939,(9879-4939)], generator=torch.Generator().manual_seed(1))

# train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=1) 
# val_loader = DataLoader(dataset=val_data, shuffle=True, batch_size=1)      
 






import torch.utils.data as data
train_data_tensor, val_data_tensor = data.random_split(dataset,[4939,(9879-4939)],generator=torch.Generator().manual_seed(1))


# now the df is an object of a custom built class with the data divided as x an y tensors
# make training and validation data objects and feed into the DataLoader()


train_loader = DataLoader(dataset=train_data_tensor, shuffle=True, batch_size=5) 
val_loader = DataLoader(dataset=val_data_tensor, shuffle=True, batch_size=5)      
 







###########################################################TRAIN THE MODEL with Gradient Descent ############################################
# The code provided creates a single model instance, and this model is updated iteratively within each epoch using batches of data. 
#                                              there is only one model throughout the training process.
# Multiple epochs are used in training to allow the model to see the entire dataset multiple times. While the model instance remains 
# the same throughout the epochs, the parameters (weights and biases) of the model are updated during each epoch based on the entire 
#                                             dataset or batches of it. Here's why multiple epochs are necessary:



criterion = nn.BCELoss()                                           # 3d data with 2 samples
# trainloader = DataLoader(dataset = train_data ,batch_size=1)           # get training data
model = logistic_regression(39)                                # Feed our model object based on data dimension!!!!
learning_rates = [.00001,.0001,.001,.01,.1,1]
#optimizer = optim.SGD(model.parameters(),lr= .1)   # Stochastic gradient descent optimizer for each learning rate
criterion = nn.BCELoss()
checkpoint_path = 'checkpoint_model.pt'              # sometimes we need to write out each epoch
checkpoint = {'epoch':None,                         # assign each epoch here 
              'model_state_dict':None,
              'optimizer_state_dict':None,
              'Loss':None} 
models = []
train_loss_list1 = []
val_loss_list2 = []

train_loss_avg = []
val_loss_avg = []

num_epochs = 10


for lr in learning_rates:
    model = logistic_regression(39)  # Initialize model for each learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr)  # Initialize optimizer with current learning rate
    
    train_loss_list1 = []  # Reset lists for each learning rate
    val_loss_list2 = []


    for epoch in range(num_epochs):
    # This updates and returns a single model 
    # This goes thru each point as it updates and optimizes weights and parameters
    # returning a single model for each epoch of  a given learning rate
    # step are the batch sizes or number of times to cover the data 
    # we use step and enumerate in order to use if logic and print
        for step, (x, y) in enumerate(train_loader): # for every batch iterations of x y in our new data 
            yhat = model(x)                          # predict a y value from all features/predictors
            loss = criterion(yhat,y)                 # calculate a CROSS ENTROPY LOSS for that point vs our predictor
            optimizer.zero_grad()                    # resets the gradient
            loss.backward()                          # creates a set of derivatives from the loss equation and solves 
            optimizer.step()                         # update the gradient descent optimizer with a model for each learning rate
             
            # checkpoints help us track progress
            checkpoint['epoch']= epoch
            checkpoint['model_state_dict']= model.state_dict()
            checkpoint['optimizer_state_dict']= optimizer.state_dict()
            checkpoint['Loss']= loss
            #print(checkpoint)
            torch.save(checkpoint,checkpoint_path)

            # the model us used on  validation and training data 
            # we make a list for every epoch
            # and then calculate and average            
            yhat2_train = model(train_data_tensor.dataset.x)          
            loss_train = criterion(yhat2_train,train_data_tensor.dataset.y)
            train_loss_list1.append(loss_train)
            if step == len(train_loader)-1:
                train_loss_avg.append(sum(train_loss_list1)/len(train_loss_list1))   
             

            yhat2_val = model(val_data_tensor.dataset.x)          
            loss_val = criterion(yhat2_val,val_data_tensor.dataset.y)
            val_loss_list2.append(loss_val)
            if step == len(train_loader)-1:
                val_loss_avg.append(sum(val_loss_list2)/len(val_loss_list2))

            #print(f""" at step {step} the training loss is {loss_train} the validation loss is {loss_val}""") 
             

    print(f""" The Avg Validation loss for learning rate {lr}is  {val_loss_avg} """)
    print(f""" The Avg Training loss for learning rate {lr} is {train_loss_avg} """)










##########################################################################################################################################

y = [tensor.item() for tensor in val_loss_avg]
y1 =[tensor.item() for tensor in train_loss_avg]

numbers = list(range(0, 4))
repeated_numbers = numbers * 6
x1 = (list(range(1, 11))) * 6
x2 = ["lr1"  * 6, "lr2"* 6, "lr3"* 6,"lr4"* 6 ,"lr5"* 6,"lr6"* 6]

from matplotlib import pyplot as plt
plt.plot(list(range(0, 60)), y)
#plt.plot(list(range(0, 60)), y1)
plt.xlabel('every ten epochs is a new learning rate')
plt.ylabel('Loss')
plt.title('Training Loss for learning rates .00001,.0001,.001,.01,.1, and 1')
plt.show()









# KEY THINGS TO CONSIDER
# CROSS ENTROPY LOSS (basically the loss for classification)

# to increase the speed consider chagning the batch size in the Data loader

# making the Data an object allows you to define which columns are x and which are y
# the the model needs to know how many features to expect
# the df converted to a tensor as x features shold reflect this 
# the Sigmoid function on the linear model is what makes it logisitic
# the Dataloader needs certain attributes to work...consider this when making objects
# steps of training is controlled by the DataLoader() num of workers argument
# performance is affected by learning rates of training and batch sizes and num of workers
# dir() allows you to examine attributes