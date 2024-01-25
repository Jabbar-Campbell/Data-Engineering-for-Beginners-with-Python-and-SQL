# League of Legions Kaggle data
# 
import pandas as pd
from torch import optim

from torch.utils.data.dataloader import DataLoader

df = pd.read_csv("C:/Users/jabba/OneDrive/Desktop/Sandbox/Python/Kaggle/high_diamond_ranked_10min.csv")
print(df.columns)                                  # blue wins is our classifier
print(df.shape)                                    # 40 features with 9879 samples
my_list = df.columns.get_values().tolist()         # list of features
my_list.remove('blueWins')                         # list except the our y predictor

# this is our actual y 
y = df['blueWins']


# convert to a tensor
import torch
import numpy as np
tensor_df = torch.Tensor(np.array(df))







## CReate a Logistic Regression Model Object
import torch.nn as nn
class logistic_regression(nn.Module):                                # our class with the objects we;d like to copy
    def __init__(self,in_size):                                      # initalize self along with any other arguments
        super(logistic_regression,self).__init__()                   # give logistic regression the features of nn.Module
        self.linear = nn.Linear(input_size= in_size,output_size=1)   # new variable linear is a model based on Linear()
    
    def forward(self,x):
        out = nn.sigmoid(self.linear(x))                            # a new function that takes the linear model of x and puts it thru the Sigmoid function
        return out





import torch 
criterion = nn.BCELoss()
dataset = (tensor_df )                                              # 3d data with 2 samples
trainloader = DataLoader(dataset = dataset ,batch_size=1)           # get training data
model = logistic_regression(40,9879)                                # Feed our model object based on data dimension!!!!
optimizer = optim.SGD(model.parameters(lr = 0.01))                  # Stochastic gradient descent optimizer
criterion = nn.BCELoss()

for epoch in range(100):
        for my_list,blueWins in trainloader:       # for every point in the sampled data of batch size 1
            yhat = model(my_list)                  # predict a y value from all features except blue wins???
            loss = criterion(yhat,y)               # calculate a CROSS ENTROPY LOSS for that point
            optimizer.zero_grad()                  # resets the gradient
            loss.backward()                        # creates a set of derivatives and solves 
            optimizer.step()                       # update the gradient descent optimizer









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
