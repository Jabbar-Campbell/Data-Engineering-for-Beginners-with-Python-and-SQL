 
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









# ########################################   DERIVATIVES #############################################################################################

# all functions/ formulas have a derivative which describes the
# slope across the entire equation/polynomial as a  rate of change 
# for example the exponential x^3 possess a derivative of 3x^2
# pytorch allows us to compute the derivative of a function with the x.backward() function where Y is the equation
# and allows us to plug in a vlaue for our newly created derivative with  the x.grad() = function where x is our value 
# its important to set the requires_grad argument to True.

q=torch.tensor(1.0,requires_grad=True) # define value in this case a scalar tensor
my_function=2q**3+q                    # define function
my_function.backward()                 # creates derivative
q.grad                                 # solves derivative with value q 


# in the case of multiple variable a each variable is set to a constant 1 and a partial derivative is
# created. hence for every variable a New Derivative is created.
# we can then set each to a given valu

u = torch.tensor(2.0, requires_grad = True) # define values some formulas may use
v = torch.tensor(1.0, requires_grad = True) # 2 or more values
my_function = u * v + (u * v) ** 2          # define fucntion
f.backward()                                # this creates 2 partial Derivatives
print("The result is ", u.grad)             # we can calulate an answer for each
print("The result is ", v.grad)             # pytorch automatically adjust which derivative


# another way to  quickly list out a tensor is with the torch.ones(length,x) command
# it list the scalar 1 as long as the length argument

a=torch.ones(length,3)
print(a)





########################################################## TRANSFORMATION #################################################################
# in the past transforming data was done by looping thru a column of data
# another why is to def a class of the data and then make functions within that class to transfrom the data 
# now there is a function/method/transformation that is inherit to this  new class
# CLASS/FUNCTIONS can do different things and be called in seq for example....

# CLASS 1
class add_mult(object):                 #add mult is a new class
    def __init__(self,addx=1,muly=1):   #class and variables must be initialize  
        self.addx=addx                  #naming convention defined
        self.muly=muly                  #naming convention defined

    def __call__(self,sample):          #Here we create a function with variables above
        x=sample[0]                     #takes an index
        y=sample[1]                     #takes an index
        x=x+self.addx                   #transform by adding 1
        y=y*self.muly                   #transform by multuplying 1
        sample=x,y                      #output variable
        return sample                   #is returned
    
# CLASS 2
class mult(object):                   #mult is a second class
    def __init__(self,muly=100):      #class and variables must be initialize  
        self.mul=mul                  #naming convention defined

    def __call__(self,sample):        #Here we create a function with variables above
        x=sample[0]                   #takes an index
        y=sample[1]                   #takes an index
        x=x*self.mul                  #transform by multplying by 100
        y=y*self.mul                  #transform by multplying by 100
        sample=x,y                    #output variable
        return sample                 #is returned

# to run transforms in series the compose() function from the transforms library in the torchvision module
# with this we can call multiple transformations for each element of a list
from torchvision import transform

My_transformation = transform.compose([add_mult(), mult()])



# QUESTION FOR ROMEL
# THIS SEEMS COMPLICATED WHY CANT WE JUST MAKE FUNCTIONS OUTSIDE OF THE CLASS AND STRING THEM TOGETHER
# CLASS ALSO CAN BE  FUNCTIONS WHICH IS CONFUSING
# I GUESS THIS WAY INSTEAD OF A FUNCION BEING PART OF THE GLOBAL ENVIRONMENT YOU CAN HAVE IT AS A PART OF THE CLASS
# SO YOU WONT HAVE TO REMEMBER WHAT FUNCTION TO CALL ON WHOM  ?????? 
# WHY NOT JUST HAVE FUNCTIONS CALLED "ADD_MULT" AND "MULT"




