# python-code-snippets
Some useful python code snippets

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
