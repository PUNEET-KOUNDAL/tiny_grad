class Value :
    def __init__(self,data,_children=() , _op = ' '):
        self.data = data
        self._prev = set(_children)
        self._op = _op 

    def __repr__(self):
        return f" >> {self.data}"
    
    def __add__(self,other):    #interally it goes like a.__add__(b)
        out = Value(self.data + other.data ,(self,other) , '+')
        return out
    
    def __mul__(self,other):
        out = Value(self.data * other.data , (self,other) , '*')
        return out
        

    







