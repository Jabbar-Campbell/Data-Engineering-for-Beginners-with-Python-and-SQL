 
# ############################################################## GETTING STARTED WITH PYTORCH   ######################################################################
# If you want to create a new Environment leave exit python into your working directory
# and then type 'python 3.11.7 -m venv Env_with_pytorch' Python version and Environment name may vary
# otherwise just use the environment you are and install it with "pip3 install torch"

 
# once installed you can import the library and start using it 
import torch

 
# we can print out the version of torch with
torch.__version__

# ############################################################## 1D Tensor #############################################################################################
# Torch has a variable type tensor that similar to a list and
# can be indexed in a similar way here we start at the first index and stop at the third.
# the stop position is NOT uncluded for what ever reason :/
a = torch.tensor([10,9,8,7])

a[1:3]

 
# tensors that we define can be added multplied 
# the dot product operation can also be used
u=torch.tensor([1,2])
v=torch.tensor([0,1])

torch.dot(u,v)

# we can inspect the data type of a tensor
a = torch.tensor([1,2,3])
a.dtype

 
# typicall the index returns another tensor
# in order to return a numerical value we use item
a[2].item



# ########################################   2D TENSORS #############################################################################################
# Once we start adding a second list
# the tensor takes on 2nd 3rd and additional dimensions
# in pytorch a tensor-table has each row as a dimension
# this is different than a non tensor table or matrix
a= torch.tensor([[0,1,1],[1,0,1]])
print(a.ndimension())
print(a.size())


# When given a data frame
# we can convert it into a tensor in the following manner
import pandas as pd
df = pd.DataFrame({'A':[11,33,22],'B':[3,3,2]})

# this 3 row 2 col table can be written as a tensor 
# where A and B are features or dimensions by transposing
torch.tensor([[11,33,22],[3,3,2]])

# if we wanted to keep the rows as dimensions then we would write
torch.tensor([[11,3],[33,3],[22,2]])

# any easier way is to just convert it using the values attribute
# of our dataframe
torch.tensor(df.values)


# Pytorch allows us to multiply tensors togetheer using
# matrix arithmetic
X = torch.tensor([[1,0],[0,1]])
Y = torch.tensor([[2,1],[1,2]])

X_times_Y = X*Y


# Pytorch allows us to multiply tensors togetheer using
# matrix arithmetic
X = torch.tensor([[1,0],[0,1]])
Y = torch.tensor([[2,1],[1,2]])

X_times_Y = X * Y
print(X_times_Y)


# in Matrix multiplication the columns of one matrix need to 
# equal the rows of the second matrix
# we can then multiply them together for example

A = torch.tensor([[0,1,1],[1,0,1]])
B = torch.tensor([[1,1],[1,1],[-1,1]])

print(torch.mm(A,B))


