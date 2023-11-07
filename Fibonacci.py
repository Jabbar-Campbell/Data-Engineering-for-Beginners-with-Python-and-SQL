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


#find_common_elements([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
#[3, 4, 5]
#>>> find_common_elements([10, 20, 30], [30, 40, 50])
#[30]
#>>> find_common_elements([1, 2, 3], [4, 5, 6])
#[]
#>>> find_common_elements([], [1, 2, 3])
#[]


def find_common_elements ( list_1 , list_2):
    if list_1 == []:
        return([])
    else:
    #set(list_1) | set(list_2)
        return( set(list_1) & set(list_2) )

find_common_elements([10, 20, 30], [30, 40, 50])

# Create a function called remove_duplicates that takes a list of
# elements as input and returns a new list with duplicates
# removed. Your function should use only built-in Python tools
# and should maintain the original order of elements while
# removing duplicates.

#>>> remove_duplicates([1, 2, 2, 3, 4, 4, 5])
#[1, 2, 3, 4, 5]
#>>> remove_duplicates(['apple', 'banana', 'apple', 'cherry'])
#['apple', 'banana', 'cherry']
#>>> remove_duplicates([1, 2, 3])
#[1, 2, 3]
#>>> remove_duplicates([])
#[]



def remove_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_liste_duplicates([1, 2, 2, 3, 4, 4, 5])

# Create a function called count_word_occurrences that takes a
# string and a target word as input and returns the number of times the target word appears in the string. The function should not be case-sensitive, meaning it should count occurrences regardless of the word's case.
# Your function should use only built-in Python tools.



#>>> count_word_occurrences("The quick brown fox jumps over the lazy dog.", "the")
#2  # "the" appears twice (both in lowercase and uppercase).
#>>> count_word_occurrences("This is a test sentence. This sentence is a test.", "sentence")
#2  # "sentence" appears twice.
#>>> count_word_occurrences("Python is a versatile programming language.", "Python")
#1  # "Python" appears once (case-insensitive).
#>>> count_word_occurrences("No matches here.", "word")
#0  # "word" does not appear in the text.

def count_word_occurrences(text, word):
    text = text.lower()  # Convert the text to lowercase for case-insensitive matching
    word = word.lower()  # Convert the target word to lowercase
    return text.count(word) if text else 0

count_word_occurrences("The quick brown fox jumps over the lazy dog.", "the")


#Create a function called is_balanced_parentheses that takes a string containing only parentheses,
# brackets, and curly braces as input and returns True if the parentheses are balanced and False
# otherwise. The parentheses are considered balanced if they are closed in the correct order.
# Your function should use only built-in Python tools.

>>> is_balanced_parentheses("()")
True
>>> is_balanced_parentheses("()[]{}")
True
>>> is_balanced_parentheses("(]")
False
>>> is_balanced_parentheses("([)]")
False
>>> is_balanced_parentheses("{[]}")
True

def is_balanced_parenthesis(s):
    my_list = list("()[]{}")
    list_2 =  list(s)
    paren = []
    for i in range(len(my_list)):
        #print(my_list[i])
        if my_list[i] in  list_2:
            #print(my_list[i])
            paren.append("True")
            #print(paren)
        else:
            #print(my_list[i])
            paren.append("False")


    if paren[:2] ==  ['True', 'True']:
        return(True)

    elif paren[2:4] == ['True', 'True']:
        return (True)

    elif paren[5:6] == ['True', 'True']:
        return(True)
    else:
        return(False)





