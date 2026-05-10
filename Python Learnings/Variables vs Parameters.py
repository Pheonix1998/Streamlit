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
    print(f"Cleaned name: {new_name}")
    print(f"Original name: {name}")
clean_name("   sImOn RiLey @   ")