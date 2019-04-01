# python-code-snippets
Some useful python code snippets

## detect python version inside python kernel
```python
import sys
sys.version_info
```

## get the class name of an object
[Getting the class name of an instance?](https://stackoverflow.com/questions/510972/getting-the-class-name-of-an-instance)
```python
s = "x"
print(s.__class__.__name__)
```

## logging
```python
import logging
logging.info("hello") #INFO:root:hello
```

## write list of strings to a file
```python
lines = ['abc', 'def']
with open('xxx.txt', 'w', encoding='utf-8') as f:
    f.writelines(map(lambda line : line+'\n', lines))
```

## count substring in a string
```python
str = "hello, goodbye, and hello"
substr = "hello"
print(str.count(substr))
```

## to remove non-ascii characters from a string
```python
s = s.encode('ascii', errors='ignore').decode()
```

## check if a string is english
https://stackoverflow.com/questions/27084617/detect-strings-with-non-english-characters-in-python
```python
def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
```

## to check the layers of a hdf5 weight file
```python
import h5py
f = h5py.File('<your-weight-file>.h5', 'r')
list(f.keys())
```

## to use IPython magic in a script file
```python
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("matplotlib qt5") # show up a window when drawing with matplotlib
```

## from two list to a list of tuples
https://stackoverflow.com/questions/2407398/how-to-merge-lists-into-a-list-of-tuples
```python
l1 = [1,2,3]
l2 = [4,5,6]
print(list(zip(l1, l2))) #[(1, 4), (2, 5), (3, 6)]
```

## convert list of tuples to two lists
https://stackoverflow.com/questions/8081545/convert-list-of-tuples-to-multiple-lists-in-python
```python
l = [(1, 2), (3, 4), (5, 6)]
z = zip(*[(1, 2), (3, 4), (5, 6)])
l1, l2 = map(list, z)
print(l1) #[1, 3, 5]
print(l2) #[2, 4, 6]
```

## list comprehension with if else
https://stackoverflow.com/questions/4406389/if-else-in-a-list-comprehension
```python
l = [-1, 3, -4, 5, 6, -9]
l = [x if x >= 0 else 0 for x in l]
print(l) #[0, 3, 0, 5, 6, 0]
```

## invert a dict mapping
```python
inv_map = {v: k for k, v in map.items()}
```

## sort a dict by key or value
To sort by key, use `key=operator.itemgetter(0)`; to sort by value, use `key=operator.itemgetter(1)`

`reverse=False` for ascending, `reverse=True` for descending
```python
import operator
d = {'math': 99, 'english': 80, 'chemistry': 67, 'biology': 88, 'physics': 93}
sorted_d = sorted(d.items(), key=operator.itemgetter(0), reverse=True)
sorted_d
```

## pretty print a dict
```python
import pprint
d = {'math': 99, 'english': 80, 'chemistry': 67, 'biology': 88, 'physics': 93}
pprint.pprint(d, width=1)
```

## random sample from a dict
```python
import random
d = {'math': 99, 'english': 80, 'chemistry': 67, 'biology': 88, 'physics': 93}
dict(random.sample(d.items(), 2))
```

## get parent class of a class
```python
<classname>.__bases__
```

Example:
```python
UnicodeDecodeError.__bases__ #(UnicodeError,)
```

## get all parent classes of a class
```python
import inspect
inspect.getmro(<classname>)
```

Example:
```python
inspect.getmro(UnicodeDecodeError)
"""
(UnicodeDecodeError,
 UnicodeError,
 ValueError,
 Exception,
 BaseException,
 object)
"""
```

## pandas: append new row
```python
import pandas as pd
df = pd.DataFrame(columns=["n", "2n", "3n"])
for i in range(3):
    df.loc[len(df)] = [i, i*2, i*3]
```

## pandas: read csv, combine all last columns
This helps when the last columns of csv file may contain ','.
```python
import csv
import pandas as pd

def to_n_columns(l, n):
    return l[:n] + [','.join(l[n:])]

with open('xxx.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = [line for line in reader]
    lines = [to_n_columns(line) for line in lines]
    
df = pd.DataFrame(lines)
```

## pandas: normalize a dataframe
```python
import pandas as pd

# df = ...
# add epsilon to ensure that nan won't appear!
df_norm = (df - df.min())/(df.max() - df.min() + np.finfo(np.float32).eps)
```

## pandas: select column names if their max values are smaller than some number
```python
import pandas as pd

# df = ...
df.loc[:,df.max() < 1e-3].columns
```

## pandas: exclude columns from dataframe
```python
import pandas as pd

# df = ...
df[df.columns.difference(['column_to_exclude'])]
```

## pandas: replace elements in a series with the mode in a rolling window
```python
from scipy.stats import mode
import pandas as pd

w_size = 5
s = pd.Series([0,0,0,1,0,0,2,2,2,2,0,1,2,2,0])
s = s.rolling(w_size).apply(lambda x: mode(x)[0], raw=False)
s[:w_size] = s[w_size]
print(s.tolist()) # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0]
```

## numpy: delete element by index/value
```python
import numpy as np
a = np.array([0,0,2,2,4,4])
# delete by index
np.delete(a, 0)
# delete by value
np.delete(a, np.where(a==0))
```

## numpy: get indices of N maximum values
https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
```python
import numpy as np
N = 3
x = np.array([4,2,1,3,5])
np.argsort(x)[-N:][::-1] # array([4, 0, 3])
```

## numpy: set output digit limitation
https://stackoverflow.com/questions/2891790/how-to-pretty-printing-a-numpy-array-without-scientific-notation-and-with-given
```python
import numpy as np
np.set_printoptions(precision=2)
```

## numpy: epsilon
https://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon
```python
print(np.finfo(float).eps)# 2.22044604925e-16
print(np.finfo(np.float32).eps)# 1.19209e-07
```
