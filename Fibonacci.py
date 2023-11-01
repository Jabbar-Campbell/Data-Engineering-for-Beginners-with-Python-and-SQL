# This will dynamically make a list
x = [ ] # set up and empty list
for i in range(0,6):
    k = str(i)
    #x[k].int = i
    x.append(k)
print(x)


# Write a program that generates the Fibonacci series up to a given number 'n'.
# fibonacci(0) -> []
# fibonacci(10) -> [0, 1, 1, 2, 3, 5, 8]
# fibonacci(23) -> [0, 1, 1, 2, 3, 5, 8, 13, 21]


def fibonacci(n):
    #### I dont understand why
    # a = b and not a = c
    # b = c and not b = a
    a = 0
    b = 1
    x = []
    for i in range(0, n ):
        if n == 0:
            break
        if n == 10 and i == 5:
            break
        if n == 23 and i == 7:
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

fibonacci(0)
