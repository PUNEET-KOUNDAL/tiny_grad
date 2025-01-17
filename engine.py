class Value:
    def __init__(self , data , _children=(), _op= ''):
        self.data =  data
        self.grad = 0 
        #internal variables used for autograd graph construction
        self.backward = lambda : None 
        self._prev = set(_children)
        self._op = _op #the op that produce this node , for graphviz / debugging / etc .

        