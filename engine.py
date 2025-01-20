class Value:
    def __init__(self , data , _children=(), _op= ''):
        self.data =  data
        self.grad = 0 
        self.backward = lambda : None 
        self._prev = set(_children)
        self._op = _op 

    def __add__(self , other ) :
        other = other if isinstance(other , Value ) else Value (other)
        out = Value (self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward


        return out 

