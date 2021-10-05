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
a = np.array([1,2,0,0,4,0])
print(a[a != 0])
```
