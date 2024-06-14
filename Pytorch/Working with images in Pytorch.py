import pandas as pd 
import os 
import torch
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader


#OS module lets us use paths
# this is a key that references each image
directory= "resources/data"
csv_file = "index.csv"
csv_path = os.path.join(directory,csv_file)

my_table= pd.read_csv(csv_path)
my_table.head()



# retrieves details on a particular image entry
# from our table via index
print("File name", my_table.iloc[1,1])
print("class", my_table.iloc[1,0])

image_name = my_table.iloc[1,1]
image_path = os.path.join(directory,image_name)



#Matplotlib lets us inspect images
my_img = Image.open(image_path)
plt.imshow(my_img, cmap = 'gray',vmin =0,vmax =255)
plt.title(my_table.iloc[1,0])                        #changes this loads a differen image
plt.show()




# we can construct a class for our image that has functions we might find useful


class my_img_class(Dataset):
    def __init__(self,csv_file,data_dir,transform = None):  #class intialized and will be refered to as self
        self.transform = transform                          #a class variable
        self.data_dir = data_dir                            #a class variable
        self.csv_file = csv_file                            #a class variable

        image_path = os.path.join(data_dir,csv_file)        #this is a local variable

        self.my_table = pd.read.csv(image_path)             #a local variable is used to make a class variable??????
        self.len = self.my_table.shape[0]                   #a class variable is used to assign another class variable??????
 
    def __len__(self):                                      # i think self dicerns the class variable from the function variable self
        return self.len
    

    def __getitem__(self,idx):                                          # when this class function is called index is supplied
        img_name= os.path.join(self.data_dir,self.my_table.iloc[idx,1]) # i think self dicerns the class variable from the function variable ie idx
        image = image.open(img_name)

        y= self.my_table.iloc[idx,0]

        if self.transform:
            image=self.transform(image)
        return image,y
    
# new class can be used with the added functions within
# its almost like cloning and environment
my_img_class.__getitem__(5)

# ASK ROMEL
# POSSIBLE REASONS FOR MAKING A CLASS WITH FUNCTIONS WITHING
#1)??
#2)???
#3)???
    

# torchvision has some image processing transformations
# such as.....transforms.CenterCrop
# compose lets us chain these transforms together
torchvision.transforms.CenterCrop()



my_tranformed_image= transforms.Compose[(transforms.CenterCrop()),
                                       (transforms.ToTensor()),
                                       (other_transforms)]



# Torchvision has some datasets available
# including the MNIST dataset
# it allows us to call on it and ....
import torchvision.datasets as dsets   

my_dataset = dsets.MNIST(root = './data',                   #....pick the root directory
                          train = False,                    #...decide weather to use the training set
                          download = True,                  # download if you dont already have the data
                          transform=transforms.ToTensor())   # convert it to a tensor

# I wonder if you can make you're own data sets and feed it in to dsets.