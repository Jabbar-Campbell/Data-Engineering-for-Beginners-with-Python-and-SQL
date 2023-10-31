# This will dynamically make a list
x=0
x = [ ] # set up and empty list
for i in range(0,6):
    k = str(i)
    #x[k].int = i
    x.append(k)
print(x)


# This will dynamically make a list
x = 0
a = 1
b = 0
n = 5
x = [ ] # set up and empty list

for i in range(0,n):
    if x == 0:
        c = a + b
        a = c
        b = a
        x.append(c)
    else:
        x != 0
        c = a + b
        a = c
        b = a
        x.append(c)
print(x)


#### I dont understand where is my 3?
a = 0
b = a
x = [ ]
def fibonacci(n):
    for i in range(0,n):
     c = a + b
     a = c
     b = a
     #c = str(c)
     x.append(c)
print(x)
