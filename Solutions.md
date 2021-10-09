#### 1. Import the numpy package under the name 'np'
```python
import numpy as np
```
#### 2. Print the numpy version and the configuration
```python
np.__version
np.show_config()
```
#### 3. Create a null vector of size 10
```python
a = np.zeros(10)
```
#### 4. How to find the memory size of any array
```python
X = np.empty((10, 10))
print("X: %d bytes" % (X.size * X.itemsize))
Y = np.ones((10, 10), dtype=np.int32)
print("Y: %d bytes" % (Y.size * Y.itemsize))
```
#### 5. How to get the documentation of the numpy add function from the command line?
```python
%run `python -c "import numpy as np; np.info(np.add)"`
```
#### 6. Create a null vector of size 10 but the fifth value which is 1
```python
a = np.zeros(10)
a[4] = 1
print(a)
```
#### 7.  Create a vector with values ranging from 10 to 49
```python
a = np.arange(10, 50)
print(a)
```
#### 8. Reverse a vector (first element becomes last)
```python
a = np.arange(0, 10)
reversed_a = np.flip(a)
print(reversed_a)
```
#### 9. Create a 3x3 matrix with values ranging from 0 to 8
```python
a = np.arange(0, 9).reshape((3, 3))
print(a)
```
#### 10. Find indices of non-zero elements from [1,2,0,0,4,0]
```python
a = np.array([1, 2, 0, 0, 4, 0])
print(np.nonzero(a))
```
#### 11. Create a 3x3 identity matrix
```python
a = np.eye(3)
print(a)
```
#### 12. Create a 3x3x3 array with random values
```python
a = np.random.random((3, 3, 3))
print(a)
```
#### 13. Create a 10x10 array with random values and find the minimum and maximum values
```python
a = np.random.random((10, 10))
amin, amax = a.min(), a.max()
print(amin, amax)
```
#### 14. Create a random vector of size 30 and find the mean value
```python
a = np.random.random(30)
print(a.mean())
```
#### 15. Create a 2d array with 1 on the border and 0 inside
```python
a = np.ones((5, 5))
a[1:-1, 1:-1] = 0
```
#### 16. How to add a border (filled with 0's) around an existing array?
```python
Solution 1:
a = np.ones((10, 10))
b = np.zeros((a.shape[0] + 2, a.shape[1] + 2))
b[1:-1, 1:-1] = a
print(b)
Solution 2:
a = np.ones((10, 10))
b = np.pad(a, 1, mode='constant', constant_values=0)
print(b)
```
#### 17. What is the result of the following expression?
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```
```python
nan
False
False
nan
True
False
```
#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
```python
a = np.diag(np.arange(1, 5), k=-1)
print(a)
```
#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern
```python
a = np.zeros((10, 10))
a[::2, ::2] = 1
a[1::2, 1::2] = 1
print(a)
```
#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element
```python
Solution 1:
Compute by yourself
(x, y, z) = (1, 5, 3)
Solution 2:
print(np.unravel_index(99, (6,7,8), order='C'))
print(np.unravel_index(99, (6,7,8))
```
#### 21. Create a checkerboard 8x8 matrix using the tile function
```python
a = np.array([[1,0],[0,1]])
b = np.tile(a, (4, 4))
print(b)
```
#### 22. Normalize a 5x5 random matrix
```python
a = np.random.random((5, 5))
a = (a - a.mean()) / a.std()
print(a)
```
#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)
```python
color = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
```
#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) 
```python
a = np.dot(np.ones((5,3), np.ones(3,2))
# Alternative solution
a = np.ones((5,3)) @ np.ones((3,2))
print(a)
````
#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. 
```python
a = np.arange(12)
a[(a > 3) & (a < 8)] *= -1
```
#### 26. What is the output of the following script?
```python
# Author: Jake VanderPlas
print(sum(range(5)), -1)
from numpy import *
print(sum(range(5)), -1)
```
```
9
10
```
#### 27. Consider an integer vector Z, which of these expressions are legal? 
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
```python
Z**Z legal
2 << Z >> 2 legal
Z <- Z legal
1j*Z legal
Z/1/1 legal
Z<Z>Z illegal
```
#### 28. What are the result of the following expressions?
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0) # floor divide
np.array([np.nan]).astype(int).astype(float)
```
```python
nan
0
array([-2.14748365e+09])
```
#### 29. How to round away from zero a float array ?
```python
a = np.random.uniform(-10, +10, 10)
# Solution 1: Readable but not so efficient
print(np.where(a > 0, np.ceil(a), np.floor(a)))
# Solution 2: More efficient
print(np.copysign(np.ceil(np.abs(a)), a))
```
#### 30. How to find common values between two arrays? 
```python
a = np.random.randint(0, 10, 10)
b = np.random.randint(0, 10, 10)
print(np.intersect1d(a, b))
```
#### 31. How to ignore all numpy warnings (not recommended)? 
```python
# Dangerous mode on
defaults = np.seterr(all='ignore')
np.geterr()
# Back to safe mode
_ = np.seterr(**defaults)
np.geterr()
# Alternative solution
 with np.errstate(all='ignore'):
    np.geterr()
```
#### 32. Is the following expressions true? 
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
```python
nan != 1j False
```
#### 33. How to get the dates of yesterday, today and tomorrow?
```python
yesterday = np.datetime64('today') - np.timedelta64(1, 'D')
today = np.datetime64('today')
tomorrow = np.datetime64('today') + np.timedelta64(1, 'D')
```
#### 34. How to get all the dates corresponding to the month of July 2016?
```python
a = np.arange('2016-07', '2016-08', dtype='datetime[D]')
print(a)
```
#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)?
```python
a = np.ones((3, 3))
b = np.ones((3, 3)) * 3
np.add(a, b, out=b)
np.divide(a, -2, out=a)
np.multiply(a, b, out=a)
```
#### 36. Extract the integer part of a random array of positive numbers using 4 different methods
```python
a = np.random.uniform(1, 5, 3)
# Solution 1
print(np.floor(a))
# Solution 2
print(a.astype(int))
# Solution 3
print(np.trunc(a))
# Solution 4
print(a - a%1)
# Solution 5
print(a // 1)
```
#### 37. Create a 5x5 matrix with row values ranging from 0 to 4
```python
# Solution 1
a = np.tile(np.arange(5), (5, 1))
print(a)
# Solution 2
a = np.zeros((5,5))
a += np.arange(5)
print(a)
```
#### 38. Consider a generator function that generates 10 integers and use it to build an array
```python
def generator():
   for x in range(10):
      yield x
a = np.fromiter(genertor(), dtype=float, count=-1)
print(a)
```
#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
```python
a = np.linspace(0,1,11,endpoint=False)[1:]
print(a)
```
#### 40. Create a random vector of size 10 and sort it 
```python
a = np.random.random(10)
a.sort()
print(a)
```
#### 41. How to sum a small array faster than np.sum?
```python
a = np.arange(3,6)
print(np.add.reduce(a))
```
#### 42. Consider two random array A and B, check if they are equal
```python
a = np.random.random((4, 5))
b = a
# Solution 1
print(np.array_equal(a, b))
# Solution 2
print(np.allclose(a, b))
```
#### 43. Make an array immutable (readonly)
```python
a = np.ones(5)
a.flags.writeable = False
a[0] = 2
```
#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
```python
a = np.random.random((10, 2))
X,Y = a[:, 0], a[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X)
print(R)
print(T)
```
#### 45. Create random vector of size 10 and replace the maximum value by 0
```python
a = np.random.random(10)
# Solution 1
a[a == a.max()] = 0
# Solution 2 np.argmax(a)
a[a.argmax()] = 0 
```
