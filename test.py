from engine import Value


a=Value(1) 
b=Value(2)
c=Value(3)


d=a+b+c
e=a*b
print(d)
print(a,b,c)
print(Value(10))
print(e)

print(a+b*c)
print(a.__add__(b).__mul__(c)) #internally it goes like a.__add__(b)

o = a.__mul__(b)
print(e)

print(e._prev)

#After implementation of backward pass
print("from here")
o._grad = 1.0
print(o._backward())