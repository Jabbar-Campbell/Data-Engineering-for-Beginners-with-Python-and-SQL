# %%
# If you want to create a new Environment leave exit python into your working directory
# and then type 'python 3.11.7 -m venv Env_with_pytorch' Python version and Environment name may vary
# otherwise just use the environment you are and install it with "pip3 install torch"

# %%
# once installed you can import the library and start using it 
import torch

# %%
# we can print out the version of torch with
torch.__version__

# %%
# Torch has a variable type tensor that similar to a list and
# can be indexed in a similar way here we start at the first index and stop at the third.
# the stop position is NOT uncluded for what ever reason :/
a = torch.tensor([10,9,8,7])

a[1:3]

# %%
# tensors that we define can be added multplied 
# the dot product operation can also be used
u=torch.tensor([1,2])
v=torch.tensor([0,1])

torch.dot(u,v)

# %%
a = torch.tensor([1,2,3])
a.dtype

# %%
# typicall the index returns another tensor
# in order to return a numerical value we use item
a[2].item


