# python-code-snippets
Some useful python code snippets

## detect python version inside python kernel
```python
import sys
sys.version_info
```

## to remove non-ascii characters from a string
```python
s = s.encode('ascii', errors='ignore').decode()
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
