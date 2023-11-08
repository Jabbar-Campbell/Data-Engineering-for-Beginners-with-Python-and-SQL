
# here we create a class called car
# create attributes for it
# those attributes can then be used in a function
# that will be methods inherent to the class

class Car:
    def __init__(self,make,model,year):
        self.make = make
        self.model = model
        self.year = year

    def display_info(self):
        return f"{self.year} {self.make} {self.model}"

# we can no create objects of class car and invoke methods by...
my_car = Car(make="Acura", model="TSX", year=2004)
my_car.display_info()






# when want to creat a new class that has the same classes attributes
# as car we can avoid a little typing when we use super ()
# we still need to intialize, and I think we want to name
# the attributes we want to take from Car I think other wise it takes
# all attributes...note display_info() is re written to include battery

class Car2(Car):
    def __init__(self,make,model,year,battery):
        super().__init__(make, model, year)
        self.battery = battery

    def display_info(self):
        return f"{self.year} {self.make} {self.model} {self.battery}"


mel_car = Car2("Tesla","Y", 20015, "electric")
mel_car.display_info()


class MaxNumberFinder:

    def __init__(self, nums):
        self.nums = nums

        # your code here

    def find_max_number(self):
        if self.nums == []
            return(None)
        else:
            return (max(self.nums))


finder = MaxNumberFinder(nums=[1,3,4,2])
finder.find_max_number()


# Create a class called Stack that implements a stack data structure.
# The stack should support the following operations:
# push(item): Pushes an item onto the top of the stack
# pop(): Removes and returns the item at the top of the stack.
# peek(): Returns the item at the top of the stack without removing it.
# is_empty(): Returns True if the stack is empty and False otherwise.
# size(): Returns the number of items in the stack.
# NOTICE HOW WE CAN REFER TO OTHER FUNCTIONS INSIDE THE CLASS
# SUCH AS is.empty2
 

class Stack:
    def __init__(self):
        self.item = []

    def push(self, item2):
        return( self.item .insert( len(self.item), item2))

    def pop(self):
        return(self.item.pop(self.item[1]))


    def peek(self):
        return (self.item[len(self.item) -1])

    def is_empty(self):
        if self.item == []:
            return(True)
        else:
            return(False)

    def size(self):
        if self.item == []:
            return(0)
        else:
            return(len(self.item)-1)

    def is_empty2(self):
        if self.is_empty() == True:
            print( "Spelled T-R-U-E")

stack = Stack()


stack.size()

stack.push(1)
stack.push(2)
stack.push(3)





print(stack.pop())  # Should print 3
print(stack.peek())  # Should print 2
print(stack.is_empty())  # Should print False
print(stack.size())  # Should print 2