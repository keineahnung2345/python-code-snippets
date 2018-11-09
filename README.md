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
