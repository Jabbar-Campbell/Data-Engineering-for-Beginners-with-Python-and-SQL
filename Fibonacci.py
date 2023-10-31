# This will dynamically make a list
x = [ ] # set up and empty list
for i in range(0,6):
    k = str(i)
    #x[k].int = i
    x.append(k)
print(x)





def fibonacci(n):
    #### I dont understand why
    # a = b and not a = c
    # b = c and not b = a
    a = 0
    b = 1
    x = []
    for i in range(0, n + 1):
        if n == 0:
            break
        if x == []:
            x.append(0)
        if x != []:
            c = a + b
            a = b
            b = c
            x.append(c)
        if i == 0:
            x.append(1)
            a = b
            b = c
        # c = str(c)
    return(x)

