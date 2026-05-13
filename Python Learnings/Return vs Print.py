def number_calculations (a,b,c):
    if a > 0 and b > 0 and c > 0:
        return a*(b+c)
    else:
        return "Not possible"
my_result = number_calculations(2,3,4)
print(my_result)

# Step 1: Calculate the raw total
def calculate_total(quantity, price):
    if quantity > 0 and price > 0:
        return (quantity * price)
    else:
        return 0

# Step 2: Add taxes
def add_tax(raw_amount, tax_rate):
    return raw_amount + (raw_amount * tax_rate)

# --- The Pipeline ---

# We can seamlessly pass the result of one function into the next!
step1_revenue = calculate_total(100, 20)       # Returns 2000
final_amount = add_tax(step1_revenue, 0.05)    # Feeds 2000 in, returns 2100
final = step1_revenue + final_amount
print(final)

d = 5
def add(a,b,c):
    if d > 0:
        return(d*(a + b + c))
    else:
        print(d+(a + b + c))
x = add(2,3,4)  # Output: 45
print(x)

def clean_name(name):
    return(name.strip().title())
z = clean_name("   john doe   ")
print(z)