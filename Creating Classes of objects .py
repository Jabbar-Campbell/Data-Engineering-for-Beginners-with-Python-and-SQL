
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