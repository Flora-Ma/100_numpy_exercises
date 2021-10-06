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
print(a[a != 0])
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

