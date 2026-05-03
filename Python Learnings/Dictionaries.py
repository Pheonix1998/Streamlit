my_dict = {'a': 10, 'b': 20, 'c': 30}
print(my_dict) # dictionaries are ordered.

my_dict_1 = {'a': 10, 'b': 20, 'c': 30, 'a': 40} 
print(my_dict_1) # duplicate keys are not allowed, the last value will be assigned to the key.

print(my_dict['a']) # accessing value using key
#print(my_dict[1]) # error

my_dict['c'] = 40 # adding new key-value pair
print(my_dict) # mutable using the keys

# Methods of dictionary
d = {'Name': 'John', 'Age': 30, 'City': 'New York'}
fetch = d.get('Name') # using get method to access value
print(fetch) # using get method to access value
print('Name' in d)
print('Country' in d)
print(d.keys()) # returns a view object that displays a list of all the keys in the dictionary.
print(d.values()) # returns a view object that displays a list of all the values in the dictionary.
print(d.items()) # returns all the keys with values from the dictionary as a list of tuples.

# looping
for key,value in d.items():
    print(key, value)

# add, remove, update
d.update({'Country': 'USA'}) # adding new key-value pair using update method
print(d)

store = d.pop('Salary', 'Not Available') # removing key-value pair using pop method, if key is not found it returns the default value.
print(store)

user = dict.fromkeys(['Name', 'Age', 'City'], 'Unknown') # creating a new dictionary with keys from a list and values set to a default value.
print(user)

# Task
task_user = {'id': 1, 'name': 'Alice', 'age': 25, 'city': 'New York', 'is_active': True}

tu = {
    k.upper():v.upper()
    for k,v in task_user.items()
    if isinstance(v, str)
}
print(tu)
