import numpy as np

# Vector is analagous to tensor in tensorflow and pytorch.
class Vector:
    def __init__(self, arr, label=None, _op='', parents=[]):
        try:
            arr = np.array(arr)
        except:
            raise ValueError("Exception has occured. Make sure the list/value contains values of numerical/int type.")
        
        if not np.issubdtype(arr.dtype, np.integer) and not np.issubdtype(arr.dtype, np.floating):
            raise TypeError("Input array must contain only integers or only floats")
        self.data = arr
        self.parents = parents
        self.op = _op
        self.label = label
        self.grad = None

    def __repr__(self) -> str:
        return "Vector(Data:{}, label:{})".format(str(self.data), str(self.label))
    
    def __str__(self):
        return str(self.data)
    
    def __add__(self,b):
        assert(type(self)==type(b))
        assert(b.data.shape==self.data.shape)
        c = Vector(self.data+b.data, _op='+', parents=[self,b])
        return c

    __radd__ = __add__ # Wouldn't actually be used.

    def __sub__(self,b):
        assert(type(self)==type(b))
        assert(b.data.shape==self.data.shape)
        c = Vector(self.data-b.data, _op='-', parents=[self,b])
        return c
    
    def __mul__(self,b):
        c = None
        parents = [self,b]
        if type(b)==Vector:
            c = Vector(self.data*b.data, _op='*', parents=parents)
        else:
            c = Vector(self.data*b, _op='*', parents=parents)
        return c
    
    def __rmul__(self,b):
        c = None
        parents = [self,b]
        if type(b)==Vector: # will not be the case, since __mul__ is called on the other object.
            c = Vector(self.data*b.data, _op='*', parents=parents)
        else:
            c = Vector(self.data*b, _op='*', parents=parents)
        return c
    
    def __truediv__(self,b):
        c = None
        parents = [self,b]
        if type(b)==Vector:
            c = Vector(self.data/b.data, _op='/', parents=parents)
        else:
            c = Vector(self.data/b, _op='/', parents=parents)
        return c
    
    def __rtruediv__(self,b):
        c = None
        parents = [self, b]
        if type(b)==Vector: # will not be the case, since __truediv__ is called on the other object.
            c = Vector(b.data/self.data, _op='1/', parents=parents)
        else:
            c = Vector(self.data/b, _op='1/', parents=parents)
        return c
    
    def __pow__(self,b):
        c = None
        parents = [self, b]
        if type(b)==Vector:
            c = Vector(self.data**b.data, _op='**', parents=parents)
        else:
            c = Vector(self.data**b, _op='**', parents=parents)
        return c
    
    def __rpow__(self,b):
        c = None
        parents = [self, b]
        if type(b)==Vector: # Won't be the case, since __pow__ is called on the other object.
            c = Vector(b.data**self.data, _op='1**', parents=parents)
        else:
            c = Vector(b**self.data, _op='1**', parents=parents)
        return c
    
    # calulates the gradient of the function wrt to the input.
    def back_prop(self, d = 1e-4, _c = 1):
        self.grad = _c
        if self.parents == [] or self.op=='': return
        elif self.op == '+':
            self.parents[0].back_prop(d, _c)
            self.parents[1].back_prop(d, _c)
        elif self.op == '-':
            self.parents[0].back_prop(d, _c)
            self.parents[1].back_prop(d, _c*-1)
        elif self.op == '*':
            try:
                self.parents[0].back_prop(d, _c*self.parents[1].data)
                self.parents[1].back_prop(d, _c*self.parents[0].data)
            except:
                self.parents[0].back_prop(d, _c*self.parents[1])
        elif self.op == '/':
            try:
                self.parents[0].back_prop(d, _c/self.parents[1].data)
                self.parents[1].back_prop(d, _c*self.parents[0].data/self.parents[1].data**2)
            except:
                self.parents[0].back_prop(d, _c/self.parents[1])
        elif self.op == '1/':
            self.parents[0].back_prop(d, -1*self.parents[1]*_c/self.parents[0].data**2)
        elif self.op == '**':
            try:
                self.parents[0].back_prop(d, _c*self.parents[1].data*self.parents[0].data**(self.parents[1].data-1))
                self.parents[1].back_prop(d, _c*self.parents[0].data**self.parents[1].data*np.log(self.parents[0].data))
            except:
                self.parents[0].back_prop(d, _c*self.parents[1]*self.parents[0]**(self.parents[1]-1))
        elif self.op == '1**':
            self.parents[0].back_prop(d, _c*self.parents[1]**self.parents[0]*np.log(self.parents[1]))


if __name__ == "__main__":
    x = Vector([1,2,3])
    y = Vector([4,5,6])
    a = 2*x+3*y
    print(a) 
    b = x*y - a
    print(b.parents)
    c = a**2
    print(c)
    f = c/3
    print(f)
    f.back_prop()
    print(x.grad) # df/dx
    print(y.grad) # df/dy