# Global variables are defined outside of a function and can be accessed anywhere in the code.
d = 5
def add(a,b,c):
    if d > 0:
        print(d*(a + b + c))
    else:
        print(d+(a + b + c))
add(2,3,4)  # Output: 45

# Local variables are defined inside a function and can only be accessed within that function.
def add(a,b,c):
    d = 5
    print(d*(a + b + c))
add(2,3,4)  # Output: 45

def clean_name(name):
    print(name.strip().title())
clean_name("   john doe   ")

# parameters are the variables that are defined in the function definition and are used to pass values to the function.
# arguments are the actual values that are passed to the function when it is called.
# variables are the names that are used to store values in the code. They can be global or local depending on where they are defined.
# functions are reusable blocks of code that perform a specific task. They can take parameters and return values.

## Parameters with local variables
def clean_name(name):
    new_name = name.strip().title().replace("@", "")  # Local variable for storage
    print(f"Cleaned name: {new_name}")
    print(f"Original name: {name}")
clean_name("   sImOn RiLey @   ")

# Global variable with parameters and local variables
case_rule = "upper"  # Global variable
def clean_name(name):
    new_name = name.strip().title().replace("@", "")  # Local variable for storage
    if case_rule == "upper":  # Accessing global variable
        new_name = new_name.upper()
    elif case_rule == "lower":
        new_name = new_name.lower()
    print(f"Cleaned name: {new_name}")
    print(f"Original name: {name}")
clean_name("   sImOn RiLey @   ")

# Positional and keyword arguments
def greet(name, greeting):
    print(f"{greeting}, {name}!")
greet("Alice", "Hello")  # Positional arguments
greet(greeting="Hi", name="Bob")  # Keyword arguments

case_rule = "upper"  # Global variable
def clean_name(first_name, last_name, country):
    if case_rule == "upper":  # Accessing global variable
        first_name = first_name.title().strip().replace("@", "").upper()  # Local variable for storage
        last_name = last_name.title().strip().replace("@", "").upper()  # Local
    elif case_rule == "lower":
        first_name = first_name.title().strip().replace("@", "").lower()  # Local
        last_name = last_name.title().strip().replace("@", "").lower()  # Local
    print(f"Cleaned name: {first_name} {last_name}")
    print(f"Original name: {first_name} {last_name}")
    print(f"Full name: {first_name} {last_name} from {country}")
clean_name("   sImOn ", "RiLey @   ", "USA") # positional arguement
clean_name(first_name="   sImOn ", last_name="RiLey @   ", country="USA") # keyword arguement

# *args and **kwargs(keyword args)
def summation (*args):
    print(sum(args))
    print(args)
    print(type(args))
summation(1,2,3)
summation(1,2,3,4)

def create_user_info(**kwargs):
    print(kwargs)
    print(type(kwargs))
create_user_info(name="John", age=30, country="USA")