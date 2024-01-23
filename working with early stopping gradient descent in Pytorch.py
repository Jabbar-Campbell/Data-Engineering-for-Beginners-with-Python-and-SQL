
######################################################### EARLY STOPPING ########################################################################
############################################################################################################################################################
#In early stopping instead of collecting all models  and their costs on validation data ie min (Model 1 | Model 2 | Model 3 | model 4)
# we do this after each iteration of gradient descenet training ie if Model 1 <= Model 2 <= Model 3.......if model 4 !<= model 3 stop!!!!!
# if the latest model is better than the cutoff we make this our new cutoff and save that model to a file
# 
# TRAINING DATA
# TEST DATA
# VALIDATION DATA
# The Training data is used to make our various models using gradient descent
# each will have its on micro parameters like BATCHSIZE and LEARNING RATE (lr)
# and then tested against the Validation DATA to for fittment in the form of a COST
# the model with the lowest cost on the Validation data is the winnner
       
citerion = nn.MSELoss()
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# Data is expressed as a class object
#  the user can enter variables w and b
# x is set but can also be made a variable of the class
class data_set(Dataset):
    def __init__(self,w,b, train = True):
        #super(data_set.self).__init__()                   #inherit parent attributes 
        self.w = torch.tensor(w,requires_grad=True)
        self.b = torch.tensor(b,requires_grad=True)
        self.x = torch.tensor.arange(-3,3,0.1.view(-1,1)) # range from -3 to 3 by increments of .1 view adds another dimension not sure why this is needed
        self.f = w*x+b
        self.y= self.f + .1*torch.randn(x.size()) 
        self.len = self.x.shape[0] 
        if train == True:
            self.y[0] = 0                                  # outlier values have been set (ideally we'd sample)
            self.y[50:55] = 20                             # outlier values have been set
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


# DataLoader takes a subset of our data acccording to batch
trainloader = DataLoader(dataset = dataset, batch_size=1) 


## we can also create a new object for linear regression
# that inherits much of the nn.Module functions from the torch.nn library
import torch.nn as nn
class LR(nn.Module):                                          # linear model with input sizet
    def __init__(self, input_size, output_size): 
        super(LR.self).__init__()                             # inherit parent attributes 
        self.linear = nn.Linear(input_size, output_size)      # thanks to inheritance we can call the "Linear" f(x) using variables

    def forward(self,x):                                      # I'm not sure what x is here????
        out = self.linear(x)
        return out
    

# with this new class we can make new objects with other features not included with Linear
    
# what we previously made a function for loss  torch.nn has MSELoss()
#   def criterion(yhat,y):                        
#        return torch.mean((yhat-y)**2)

criterion = nn.MSELoss()

epochs = 10
learning_rates = [.00001,.0001,.001,.01,.1,1]
validation_error = torch.zeros(len(learning_rates))
models = []
min_loss = 100 #our early stop value for validation loss/cost


checpoint_path = 'checkpoint_model.pt'              # sometimes we need to write out each epoch
checkpoint = {'epoch':None,                         # assign each epoch here 
              'model_state_dict':None,
              'optimizer_state_dict':None,
              'Loss':None} 


# Training with EARLY STOPPING
from torch import optim
for i,learning_rate in enumerate(learning_rates):
    model= LR[1,1]
    optimizer = optim.SGD(model.parameters(), lr = learning_rate )

    for epoch in range(epochs): 
        # 60 models are being generated one for each point
        for x,y in trainloader:       # for every point in the sampled data of batch size 1
            yhat = model(x)           # predict a y value 
            loss = criterion(yhat,y)  # calculate a loss for that point
            optimizer.zero_grad()     # resets the gradient
            loss.backward()           # creates a set of derivatives 
            optimizer.step()          # solves each/all derivative  at that point
            
            # if the loss is greater we go to the next model
            loss_train = criterion(model(trainloader.x),trainloader.y).item()
            loss_val= criterion(model(valdata.x),valdata.y).item()
            if loss_val < min_loss:  
                value=epoch
                min_loss = loss_val
                torch.save(model.state_dict(),'filename_best_model.pt')

    # for each epoch the model is compared
    # to actual data and appended to a list
    yhat2 = model(dataset.x)          
    loss = criterion(yhat,dataset.y)
    validation_error[i]= loss.item()
    models.append(model)

    # In each epoch the model is compared to the validation data
    # the model and its cost are appended to a list
    yhat3 = model(valdata.x)         
    loss = criterion(yhat,valdata.y)
    validation_error[i]= loss.item()
    models.append(model)

    checkpoint['epoch']= epoch
    checkpoint['model_state_dict']= model.state_dict()
    checkpoint['optimizer_state_dict']= optimizer.state_dict()
    checkpoint['loss']= loss
    checkpoint['epoch']= epoch
    torch.save(checkpoint,checpoint_path)


# we load the state dictionary and the loss and resume if need start and stop at a certain iteration
optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['Loss']

