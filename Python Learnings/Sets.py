analytics_tools = {"A", "E", "B", "C"}
analytics_tools.add("D")
print(analytics_tools)

a = {"A", "B", "C"}
a. update(["D", "E"],("F", "G"))
print(a)

#a.remove("X")
#print(a)

a.discard('X')
a.pop()
print(a)

# Mathematical Operators
a = {"A", "B", "C"}
b = {"B", "C", "D"}
print(a.union(b))
print(a.intersection(b))
print(a.symmetric_difference(b))
print(a.issubset(b))

new_a = {1,2,3}
new_b = {1,2,3,4}
print(new_b.issuperset(new_a))

new_c = {'a', 'b', 'c'}
new_d = {'x', 'y', 'z'}
print(new_c.isdisjoint(new_d))