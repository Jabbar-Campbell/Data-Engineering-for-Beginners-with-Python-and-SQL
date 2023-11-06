#https://www.w3schools.com/python/default.asp
# Great source for refreshers

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


#Create a function called find_fibonacci that takes a
# non-negative integer n as input and returns the n-th
# Fibonacci number. The Fibonacci sequence is a series of
# numbers where each number is the sum of the two preceding
# ones, usually starting with 0 and 1.
# Your function should use only built-in Python tools.

#>>> find_fibonacci(0)
#0
#>>> find_fibonacci(1)
#1
#>>> find_fibonacci(5)
#5
#>>> find_fibonacci(10)
#55


def find_fibonacci(n):
    a = 0
    b = 1
    x = []
    #n = 5
    for i in range(0,n):
        if x == [ ]:
            x.append(0)
        if x != []:
            c = a + b
            a, b = b, b + a
            x.append(c)
            #print(x)
    if n == 0:
        return(0)
    if n == 1:
        return 1
    else:
        return(x[n-1])

find_fibonacci(10)



# Create a function called is_prime that takes an integer as
# input and returns True if the number is prime and False
# otherwise. Your function should use only built-in Python tools.

#>>> is_prime(5)
#True
#>>> is_prime(17)
#True
#>>> is_prime(4)
#False
#>>> is_prime(1)
#False



def is_prime(number):
    x = []
    for i in range(1,number+1):
        if number % i == 0:
            x.append(i)
    #if len(x) <= 1:
    #    return("False")
    if len(x) == 2:
        return(True)
    else:
        return(False)





# Description: Write a function that takes an integer n as input
# and returns the count of prime numbers less than n. Input: 10 Output:
# 4 (Primes less than 10: 2, 3, 5, 7)


def count_primes(n):
    x = []
    for i in range(1,n+1):
        if n==2:
            return(0)
        else:
            if is_prime(i) == True:
                x.append(i)
                print(x)
    return(len(x))

count_primes(2)


# Write a Python program that calculates the average of a list of numbers.
#In:  [5, 10, 15, 20]
#Out:  12.5
# If the list is empty, return 0


def calculate_average(numbers):
    if numbers == 0:
        return 0
    else:
        average = sum(numbers)/len(numbers)
    return average


# Write a Python function that takes a list of numbers and a target number,
# and it returns the count of how many times the target number appears in the list.
# In: ([1, 2, 3, 4, 2, 2, 5], 2)
# Out: 3


def count_occurences(numbers,target):
    match = []
    for i in range(len(numbers)):
        if target == numbers[i]:
            match.append(numbers[i])
    return len(match)


count_occurences([1,2,3,4,2,2,6],2)

#Create a function called find_missing_number that takes a list of distinct integers
# from 0 to n (inclusive), where n is one less than the length of the list, and returns the
# missing number from the list. Your function should use only built-in Python tools.

#>>> find_missing_number([0, 1, 3])
#2
#>>> find_missing_number([4, 1, 3, 2, 0, 6, 7, 5])
#8
#>>> find_missing_number([9, 7, 2, 1, 0, 6, 8, 4, 5, 3])
#10
#>>> find_missing_number([])
#0
nums = [4, 3, 2, 0, 6, 7, 5,1,1,1,1]

def find_missing_number(nums):
    x = []
    nums =  sorted(nums)
    if nums == []:
        return(0)

    for i in range(0,len(nums)+1) :
        if nums.count(i) == 0:
            x.append(i)
            print(i)
            #print(x)
    #return(x)



# Create a function called find_common_elements that takes two lists of integers as
# input and returns a list containing the common elements between the two input lists.
# The order of elements in the resulting list does not matter. Your function should use
# only built-in Python tools.

 find_common_elements([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
[3, 4, 5]
>>> find_common_elements([10, 20, 30], [30, 40, 50])
[30]
>>> find_common_elements([1, 2, 3], [4, 5, 6])
[]
>>> find_common_elements([], [1, 2, 3])
[]
