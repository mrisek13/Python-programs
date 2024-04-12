# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:27:02 2021

@author: User
"""
##PRVI
def quicksort(niz):
    if len(niz) <= 1:
        return niz
    pivot = niz[len(niz) // 2]
    left = [x for x in niz if x < pivot]
    middle = [x for x in niz if x == pivot]
    right = [x for x in niz if x > pivot]
    return quicksort(left) + middle + quicksort(right)
print(quicksort([30,6,31,10,1,8,1]))

#Basic data types: integer, float, boolean, string


##DRUGI - cijeli broj ili s pomocnom tockom
# integer, float
x = 5
print(type(x)) # prints "<class 'int'>"
print(x)       # prints 3

##TRECI - osnovne operacije s brojevima
print(x+10)   # Addition
print(x-10)   # Subtraction
print(x*10)   # multiplication
print(x**3)  # Exponentiation
x += 1      # Adding 1 to x
print(x)  
x *= 3  # Multipliation of x with 3
print(x) 

##CETVRTI ili broj s pomocnom tockom
y = 5.7
print(type(y)) # prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) 

##PETI
#Boolean ili logički:
t = True
f = False
print(type(t)) # "<class 'bool'>"
print(t and f) # logic AND
print(t or f)  # logic OR
print(not t)   # logic NOT
print(t != f)  # logic XOR

##SESTI
# String
hello = 'hello'    # single quotes
world = "world"    # or in double quotes
print(hello)

##SEDMI
print(len(hello))  # string length
hw = hello + ' ' + world  # concatination
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # c++ sprintf stile
print(hw12)  # prints "hello world 12"

# Containers: lists, dictionaries, sets, and tuples

##OSAM
# A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:
xs = [5, 7, 2]    # Create a list
print(xs, xs[2])  
print(xs[-1])     # Negative indices count from the end of the list
xs[2] = 'abc'     # Lists can contain elements of different types
print(xs)  

##DEVET
xs.append('efg')  # Add a new element to the end of the list
print(xs)         
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)     

##DESET
# Slicing ili odvajanje-rezanje
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # from index 2 to the end
print(nums[:2])           # from the start to index 2 (exclusive)
print(nums[:])            # whole list
print(nums[:-1])          
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)   

##11

#loops
#  loop over the elements of a list :
zivotinje = ['macka', 'konj', 'krava']
for z in zivotinje:
print(z)
    
##12
# enumerate - gives indices of corresponding list elements
zivotinje = ['macka', 'konj', 'krava']
for idx, z in enumerate(zivotinje):
print('#%d: %s' % (idx + 1, z)) 
          
##13
# another example
b = [0, 1, 2, 3, 4]
b2 = []
for x in b:
    b2.append(x ** 2)
print(b2) 

##14
# we can put that in one line:
b = [0, 1, 2, 3, 4]
b2 = [x ** 2 for x in b]
print(b2)

##15
# statement if can also be used in this context
b = [0, 1, 2, 3, 4]
parni_b2 = [x ** 2 for x in b if x % 2 == 0]
print(parni_b2)

##16 Tuples A tuple is an (immutable) ordered list of values.
t = (5, 6)        # Create a tuple
print(type(t))    #  "<class 'tuple'>"

##17
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

#NUMPY

#Numpy is the core library for scientific computing in Python. It is implemented in C, and provides a high-performance multidimensional array object, and tools for working with these arrays.

#A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

#We can initialize numpy arrays from nested Python lists, and access elements using square brackets:
    
##18
import numpy as np

a = np.array([1, 2, 3])   # Create array of rank 1
print(type(a))            # "<class 'numpy.ndarray'>"
print(a.shape)            # "(3,)"
print(a[0], a[1], a[2])   # "1 2 3"
a[0] = 5                  # Change element
print(a)                  # "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Rank 2 array
print(b.shape)                     
print(b[0, 0], b[0, 1], b[1, 0])   
b

##19
import numpy as np

a = np.zeros((2,2))   # array of all zeros
print(a)              
                     
b = np.ones((1,2))    # array of all ones
print(b)              

c = np.full((2,2), 7)  # constant array
print(c)              
                       
d = np.eye(2)         # 2x2 identity matrix
print(d)            

e = np.random.random((2,2))  # array filled with random values (from 0 to 1, uniform distribution)
print(e)  

##20
# arange, the same as range but created numpy array
np.arange(3)
np.arange(3.0)
np.arange(3,7)
np.arange(3,7,2)
np.linspace(2.0, 3.0, num=5)  # formira niz datog broja tocaka jednakih rastojanja u datom intervalu
np.linspace(2.0, 3.0, num=5, endpoint=False)
np.linspace(2.0, 3.0, num=5, retstep=True)

#21
#Slicing
import numpy as np

# rang 2 array, shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# isecak: prva dva reda, i kolone 1 i 2 (druga i treca)
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# Isecak ne formira novi niz u memoriji! Ako menjamo isecak, menja se i original! 
print(a[0, 1])   
b[0, 0] = 77     
print(a[0, 1]) 

##22
# integer indexing
import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

red_r1 = a[1, :]    # drugi red niza a, ranga 1!
red_r2 = a[1:2, :]  # drugi red niza a, ranga 2!
print(red_r1, red_r1.shape)  
print(red_r2, red_r2.shape)  

# Isto sa kolonama:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape) 
print(col_r2, col_r2.shape)  

##23
# Boolean indexing - often very useful
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  

print(bool_idx)    

print(a[bool_idx])  # Returns only the elements greated than 2!

# In one line:
print(a[a > 2]) 

##24
# Data types in an array
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # "int64"

x = np.array([1.0, 2.0])
print(x.dtype)             # "float64"

x = np.array([1, 2], dtype=np.float64)   # Force a particular datatype
print(x.dtype)  

##25
#Basic mathematical functions operate elementwise on arrays
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# addition
print(x + y)
print(np.add(x, y))

# subtraction
print(x - y)
print(np.subtract(x, y))

# multiplication (elementwise!)
print(x * y)
print(np.multiply(x, y))

# division
print(x / y)
print(np.divide(x, y))

# square root
print(np.sqrt(x))

##26
# .dot is used for vector and matrix multiplication
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# dot product
print(v.dot(w))
print(np.dot(v, w))

# vector and amtrix product
print(x.dot(v))
print(np.dot(x, v))

# Product of thw matricies
print(x.dot(y))
print(np.dot(x, y))

##27
# Some useful functions
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # sum of all the elements
print(np.sum(x, axis=0))  # sum of all the columns
print(np.sum(x, axis=1))  # sum of the rows

##28
x = np.array([[1,2], [3,4]])
print(x)    
print(x.T)  # transpose

# transposing a rank 1 array is not possible
v = np.array([1,2,3])
print(v)    
print(v.T) 

#Zadaci:

#1.Pronaci element u nizu koji je najblizi datom skalaru.
#2.Formirati 10x10 array sa slucajnim elementima i pronaci minimalnu i maksimalnu vrijednost.
#3.Pronaci indekse elemenata koji nisu nula kod niza [1,2,0,0,4,0].
#4.Formirati slucajni vektor velicine 30 i pronaci srednju vrijednost elemenata.
#5.Pronaci proizvod proizvoljne 5x3 matrice i proizvoljne 3x2 matrice.

# Plotting - nastavak!

