# Global variables
a = 5
b = 10
def add(a, b):
    """Returns the sum of a and b."""
    return a + b
print(add(a, b))  # Output: 15

# Function with for loop
def print_values():
    for i in range(5):
        print(i, end=' ')
print_values()  # Output: 0 1 2 3 4

def clean_name(name):
    print(name.strip().title())
clean_name("   john doe   ")  # Output: "John Doe"

def add (x, y):
    print(x + y)
add(3, 4)  # Output: 7