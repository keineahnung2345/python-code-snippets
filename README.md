# python-code-snippets
Some useful python code snippets

## alias python3 as python
[How to make 'python' program command execute Python 3?](https://askubuntu.com/questions/320996/how-to-make-python-program-command-execute-python-3)
```sh
vim ~/.bash_aliases # add "alias python=python3"
source ~/.bash_aliases
```

## get script's directory
[How can I find script's directory? [duplicate]](https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory)
```python
import os
print(os.path.dirname(os.path.realpath(__file__)))
```

## round to specific decimals
[How to round to 2 decimals with Python?](https://stackoverflow.com/questions/20457038/how-to-round-to-2-decimals-with-python)
```python
import math

def round_down(n, d=2):
    d = int('1' + ('0' * d))
    return math.floor(n * d) / d

def round_up(n, d=2):
    d = int('1' + ('0' * d))
    return math.ceil(n * d) / d
```

## read input until EOF
[How to read user input until EOF?](https://stackoverflow.com/questions/21235855/how-to-read-user-input-until-eof/36237166)
```python
from sys import stdin

for line in stdin:
    print(line)
```

## detect python version inside python kernel
```python
import sys
sys.version_info
```

## check object size

```python
import sys

s = "abcdef"
sys.getsizeof(s) #55
```

## check object having attribute or not
[How to know if an object has an attribute in Python](https://stackoverflow.com/questions/610883/how-to-know-if-an-object-has-an-attribute-in-python)
```python
import string
hasattr(string, "lower") # False
```

## get current function name

[Determine function name from within that function (without using traceback)](https://stackoverflow.com/questions/5067604/determine-function-name-from-within-that-function-without-using-traceback)
```python
import inspect

def dummy_function():
    print(inspect.currentframe().f_code.co_name) 

class dummy_class:
    def dummy_method(self):
        print(inspect.currentframe().f_code.co_name)

dummy_function() # dummy_function

c = dummy_class()
c.dummy_method() # dummy_method
```

## check file exist
```python
import os

os.path.exists('xxx.txt')
```

## check file size
```python
import os
statinfo = os.stat('xxx.txt')
print(statinfo) # os.stat_result(st_mode=33188, st_ino=6911355, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=26571, st_atime=1565929634, st_mtime=1565346343, st_ctime=1565929634)
print(statinfo.st_size) # 26571 # the unit is byte
```

## check directory size
[Calculating a directory's size using Python?](https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python)
```python
from pathlib import Path

root_directory = Path('.')
print(sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file() ))
```

## get directory name & file name from full path
[Extract a part of the filepath (a directory) in Python](https://stackoverflow.com/questions/10149263/extract-a-part-of-the-filepath-a-directory-in-python)

```python
import os

fname = "/xxx/yyy/zzz/log.txt"
os.path.dirname(fname) #'/xxx/yyy/zzz'
os.path.basename(fname) #'log.txt'
```

## get file name without extension
[How to get the filename without the extension from a path in Python?](https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python)
```python
import os

fname = "/xxx/yyy/zzz/log.txt"
os.path.splitext(fname)[0] #'/xxx/yyy/zzz/log'
os.path.splitext(os.path.basename(fname))[0] #'log'
```

## mkdir -p
[How can I safely create a nested directory?](https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory)
```python
import os
# this works even when 'first_layer' doesn't exist
os.makedirs("first_layer/second_layer")
# succeeds even if directory exists
os.makedirs("first_layer/second_layer", exist_ok=True)
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

## read file with chinese in Windows
use `utf-8-sig` encoding so that the first line won't be prefixed with `\ufeff`
```python
with open('stop_words.txt', 'r', encoding='utf-8-sig', errors='ignore') as f:
    print(f.readlines())
```

## write file with chinese in Windows
[Python3 UnicodeDecodeError with readlines() method](https://stackoverflow.com/questions/35028683/python3-unicodedecodeerror-with-readlines-method)
```python
with open(fname, 'w', errors='ignore') as f:
    f.writelines(lines)
```

## check if a variable exists
[How do I check if a variable exists?](https://stackoverflow.com/questions/843277/how-do-i-check-if-a-variable-exists)

To check if `myVar` is one of local variables:
```python
if 'myVar' in locals():
    # myVar exists
```

To check if `myVar` is one of global variables:
```python
if 'myVar' in globals():
  # myVar exists
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

## string to byte array and vice versa

[Best way to convert string to bytes in Python 3?](https://stackoverflow.com/questions/7585435/best-way-to-convert-string-to-bytes-in-python-3)

```python
s = "12345"
b = str.encode(s) # b'12345'
type(b)           # <class 'bytes'>
s = b.decode()    # '12345'
type(s)           # <class 'str'>
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

## find Nth occurrence of a substring in a string
[Python | Ways to find nth occurrence of substring in a string](https://www.geeksforgeeks.org/python-ways-to-find-nth-occurrence-of-substring-in-a-string/)

In order:
```python
s = "a/b/c/d/e"
val = -1
for i in range(4):
    val = s.find('/', val+1)
    print(val)
"""
1
3
5
7
"""
```

In reverse order:
```python
s = "a/b/c/d/e"
val = len(s)
for i in range(4):
    val = s.rfind('/', 0, val)
    print(val)
"""
7
5
3
1
"""
```

In a function:

[Find the nth occurrence of substring in a string](https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string)
```python
def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

fname = "a_b_c_d_e_f.txt"
discard_cnt = 3 # discard 3 "_"
print(fname[:find_nth(fname, "_", fname.count("_")-(discard_cnt-1))]) # "a_b_c"
```


## string range from a to z
[Python: how to print range a-z?](https://stackoverflow.com/questions/3190122/python-how-to-print-range-a-z)
```python
import string
print(string.ascii_lowercase)
# 'abcdefghijklmnopqrstuvwxyz'
```

## string to list
[How to convert a string to a list in Python?](https://stackoverflow.com/questions/5387208/how-to-convert-a-string-to-a-list-in-python/5387227)
```python
s = "abc"
print(list(s)) # ['a', 'b', 'c']
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

## to show image in jupyter notebook
```python
import cv2
import matplotlib.pyplot as plt
# load image using cv2....and do processing.
image = cv2.imread('xxx.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# as opencv loads in BGR format by default, we want to show it in RGB.
plt.show()
```

## list: pop, remove, clear
```python
l = ['a', 'b', 'c', 'd']
l.pop(1) #'b'
print(l) #['a', 'c', 'd']
l.remove('d')
print(l) # ['a', 'c']
l.clear()
print(l) # []
```

## get N largest values from a list
[Python: take max N elements from some list](https://stackoverflow.com/questions/4215472/python-take-max-n-elements-from-some-list)
```python
import heapq

print(heapq.nlargest(3, [10, 5, 3, 8, 4, 2]))
# [10, 8, 5]
```

## max index and value of a list
[Pythonic way to find maximum value and its index in a list?](https://stackoverflow.com/questions/6193498/pythonic-way-to-find-maximum-value-and-its-index-in-a-list)
```python
l = [5,3,8,1,9]

import operator
index, value = max(enumerate(l), key=operator.itemgetter(1)) # 4,9
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

## max among a list of tuple
[Find the maximum value in a list of tuples in Python [duplicate]](https://stackoverflow.com/questions/13145368/find-the-maximum-value-in-a-list-of-tuples-in-python)
```python
l = [('a', 153), ('b', 827), ('c', 961)]
print(max(l, key=lambda x : x[1])[0]) # 'c'
```

## add two tuples element-wise
[Python element-wise tuple operations like sum](https://stackoverflow.com/questions/497885/python-element-wise-tuple-operations-like-sum)
```python
import operator

a = (1,2,3,4)
b = (4,3,2,1)

print(tuple(map(operator.add, a, b))) # (5, 5, 5, 5)
print(tuple(map(operator.sub, a, b))) # (-3, -1, 1, 3)
```

## list comprehension with if else
https://stackoverflow.com/questions/4406389/if-else-in-a-list-comprehension
```python
l = [-1, 3, -4, 5, 6, -9]
l = [x if x >= 0 else 0 for x in l]
print(l) #[0, 3, 0, 5, 6, 0]
```

## nested list comprehension
[List comprehension on a nested list?](https://stackoverflow.com/questions/18072759/list-comprehension-on-a-nested-list)
```python
l = [['40', '20', '10', '30'], ['20', '20', '20', '20', '20', '30', '20'], ['30', '20', '30', '50', '10', '30', '20', '20', '20'], ['100', '100'], ['100', '100', '100', '100', '100'], ['100', '100', '100', '100']]
l = [[int(y) for y in x] for x in l]
print(l)
"""
[[40, 20, 10, 30], [20, 20, 20, 20, 20, 30, 20], [30, 20, 30, 50, 10, 30, 20, 20, 20], [100, 100], [100, 100, 100, 100, 100], [100, 100, 100, 100]]
"""
```

## count the occurrence of elements in a list
```python
from collections import Counter
count = Counter(['apple','red','apple','red','red','pear'])
print(dict(count))
```

## flatten list of lists to a list
```python
ll = [[0, 1, 3, 4, 5], 
      [0, 1, 2, 3], 
      [0, 3, 4, 5, 6]]

[e for l in ll for e in l]
```

## find duplicate elements in a list
[Identify duplicate values in a list in Python](https://stackoverflow.com/questions/11236006/identify-duplicate-values-in-a-list-in-python)
```python
from collections import Counter
mylist = [20, 30, 25, 20]
print([k for k,v in Counter(mylist).items() if v>1]) # [20]
```

## unpack list, only get the firsk n elements
[unpack the first two elements in list/tuple](https://stackoverflow.com/questions/11371204/unpack-the-first-two-elements-in-list-tuple)
```python
a, b, *_ = [1, 2, 3, 4, 5, 6, 7]
```

## random: random float in a given range
[How to get a random number between a float range?](https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range)
```python
import random

random.uniform(1.5, 1.9)
# 1.8892901892967993
```

## weighted version of random.choice
[A weighted version of random.choice](https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice)
```python
import numpy as np

np.random.choice([1,2,3], 1, p=[0.45,0.45,0.1])[0]
```

## random split a list with a given ratio
```python
import random

data = list(range(500))
ratio = 0.9
random.shuffle(data)
train_data = data[:int(len(data) * ratio)]
test_data = data[int(len(data) * ratio):]
```

## remove all occurrences of an element in a list
```python
l = [1,2,3,2,2,2,3,4]
print(list(filter(lambda x: x != 2, l))) # [1, 3, 3, 4]
```

## list copy
This will give wrong result, updating `l2` will also updates `l`:
```python
l = [1,2,3]
l2 = l
l2[0] += 1
print(l)  # [2,2,3]
print(l2) # [2,2,3]
```
This will give correct result, updating `l2` won't updates `l`:
```python
l = [1,2,3]
l2 = l.copy()
l2[0] += 1
print(l)  # [1,2,3]
print(l2) # [2,2,3]
```

## invert a dict mapping
```python
inv_map = {v: k for k, v in map.items()}
```

## reduce
```python
from functools import reduce
arr = [1, 2, 3, 4, 5]
reduce(lambda x, y : x+y, arr)
```

## sort a dict by key or value
To sort by key, use `key=operator.itemgetter(0)`; to sort by value, use `key=operator.itemgetter(1)`

`reverse=False` for ascending, `reverse=True` for descending
```python
import operator
d = {'math': 99, 'english': 80, 'chemistry': 67, 'biology': 88, 'physics': 93}
sorted_d = sorted(d.items(), key=operator.itemgetter(0), reverse=True)
#or using lambda function, this does the same as above
#sorted_d = sorted(d.items(), key=lambda x : x[0], reverse=True)
sorted_d
```

[How do I sort a dictionary by value?](https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value)

python dict preserves insertion order(Python3.7+)!

```python
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
# {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}

dict(sorted(x.items(), key=lambda item: item[1]))
# {0: 0, 2: 1, 1: 2, 4: 3, 3: 4}
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

## convert all values of a dict to 1
```python
d = {0:1, 1:4, 2:7, 3:1, 4:9, 5:2}

print(dict(zip(d.keys(), [1]*len(d))))
```

## dict.get
Use `dict.get` so that there won't be an error when the key doesn't exist in the dict.
```python
d = {'a': 1}
d.get('a') # 1
d.get('b') # None
```

## defaultdict
```python
from collections import defaultdict

# list
d = defaultdict(list)
d['a'].append(1)
print(d) #defaultdict(<class 'list'>, {'a': [1]})

# int
d2 = defaultdict(lambda : 0)
d2['a'] += 1
print(d2) #defaultdict(<function <lambda> at 0x1106c6e18>, {'a': 1})

# int
d3 = defaultdict(int)
d3['a'] += 1
print(d3) #defaultdict(<class 'int'>, {'a': 1})

# iterate
for k, v in d.items():
    print(k, v)
```

## defaultdict of defaultdict
```python
from collections import defaultdict

d = defaultdict(lambda: defaultdict(int))
```

## sum up two dict
[Merge and sum of two dictionaries](https://stackoverflow.com/questions/10461531/merge-and-sum-of-two-dictionaries)
```python
d1 = {'a': 100, 'b':200, 'c':400}
d2 = {'a': 50, 'b': 90, 'c': -100}
d_sum = {k: d1.get(k, 0) + d2.get(k, 0) for k in set(d1) | set(d2)}
```

## add list's elements into a set
[Python: how to add the contents of an iterable to a set?](https://stackoverflow.com/questions/4045403/python-how-to-add-the-contents-of-an-iterable-to-a-set)
```python
s = set()
l = [1,3,6,3,2]
s.update(l)
print(s) # set([1, 2, 3, 6])
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

## format string, limit floats
[Limiting floats to two decimal points](https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points)
```python
f = 123.456789
print("%.2f" % f)
# 123.46
print("{0:.2f}".format(f))
# 123.46
```

## format string, zero padding
[How to pad zeroes to a string?](https://stackoverflow.com/questions/339007/how-to-pad-zeroes-to-a-string)
```python
i = 5
"{:06d}.txt".format(i) #'000005.txt'
```

## format string, space padding
[How to pad a numeric string with zeros to the right in Python?](https://stackoverflow.com/questions/40999973/how-to-pad-a-numeric-string-with-zeros-to-the-right-in-python)
```python
i = 123
j = 45
print("[{:> 4d} {:> 4d}] message".format(i, j)) # [ 123   45] message
```

## partially format string
[partial string formatting](https://stackoverflow.com/questions/11283961/partial-string-formatting)
```python
from functools import partial

fname = "{s}_{t}_{i}.txt".format
fname_p = partial(fname, s=0, t=123)
fname_p(i="pca")
```

## json: read and write
[Python JSON](https://www.w3schools.com/python/python_json.asp)
```python
import json

data = {"name": ["John", "Mary", "Kevin"],
        "area": ["London", "Munich", "Berlin"],
        "age":  [33, 56, 44]}

with open("db.json", "w") as f:
    json.dump(data, f, indent=4)

with open("db.json", "r") as f:
    data = json.load(f)

print(data)
```

## argparse: add positional argument and optional argument, argument with default value
[Python argparse: default value or specified value](https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value)
```python
import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument("pos1")
parser.add_argument("-o1", "--optional1", dest="o1")
# remove "--optional1", and we don't need 'dest="o2"!'
parser.add_argument("-o2", help="optional argument 2")
parser.add_argument("-i1", dest="i1", type=int, default=0)
parser.add_argument("-b1", action=store_true, help="pass -b1 will make b1 be true, otherwise it's false")
args = parser.parse_args()
pos1 = args.pos1
o1 = args.o1
o2 = args.o2
i1 = args.i1
```

## pandas: create dataframe with only column names

[Pandas create empty DataFrame with only column names](https://stackoverflow.com/questions/44513738/pandas-create-empty-dataframe-with-only-column-names)

```python
df = pd.DataFrame(columns=['A','B','C','D','E','F','G'])
```

```
Empty DataFrame
Columns: [A, B, C, D, E, F, G]
Index: []
```

## pandas: get row count
[How do I get the row count of a Pandas DataFrame?](https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe)
```python
len(df.index) # slightly faster
df.shape[0]
```

## pandas: append new row
[Python – Pandas dataframe.append()](https://www.geeksforgeeks.org/python-pandas-dataframe-append/)
```python
import pandas as pd
df = pd.DataFrame(columns=["n", "2n", "3n"])
for i in range(3):
    df.loc[len(df)] = [i, i*2, i*3]
```

```python
import pandas as pd
df = pd.DataFrame(columns=["n", "2n", "3n"])
df = df.append({'n': 1, '2n': 2, '3n': 3}, ignore_index=True)
```

## pandas: slice
[Python pandas slice dataframe by multiple index ranges](https://stackoverflow.com/questions/39393856/python-pandas-slice-dataframe-by-multiple-index-ranges)
```python
df.iloc[np.r_[10:12, 25:28]]
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

## pandas: get max value and its index in a column
[How to find the max value of a pandas DataFrame column in Python](https://www.kite.com/python/answers/how-to-find-the-max-value-of-a-pandas-dataframe-column-in-python)
```python
df['col1'].idxmax()
df['col1'].max()
```

## pandas: sort by columns
[how to sort pandas dataframe from one column](https://stackoverflow.com/questions/37787698/how-to-sort-pandas-dataframe-from-one-column)

The followings are equivalent:
```python
df = pd.DataFrame({"score": np.random.randint(0,100,100)})
df.sort_values(by=["score"], ignore_index=True)
```

```python
df = pd.DataFrame({"score": np.random.randint(0,100,100)})
df = df.sort_values(by=["score"])
df.reset_index(drop=True)
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

## pandas: export dataframe with chinese to csv, using correct encoding
[使用python处理中文csv文件，并让excel正确显示中文（避免乱码）](https://blog.csdn.net/xw_classmate/article/details/51940430)
```python
df.to_csv('xxx.csv', index=False, encoding='utf-8-sig')
```

## pandas: append to a csv file
[How to add pandas data to an existing csv file?](https://stackoverflow.com/questions/17530542/how-to-add-pandas-data-to-an-existing-csv-file)
```python
df.to_csv('xxx.csv', mode='a', header=False, index=False)
df.to_csv('xxx.csv', mode='a', header=False, index=False)
```

## pandas: to csv, set float format
[float64 with pandas to_csv](https://stackoverflow.com/questions/12877189/float64-with-pandas-to-csv)
```python
df.to_csv('xxx.csv', 
    float_format='%.2f',
    index=False, header=True)
```

## pandas: one dataframe has a column of type list, to filter out specific elements in the lists in this dataframe
[Find element's index in pandas Series](https://stackoverflow.com/questions/18327624/find-elements-index-in-pandas-series)
```python
stop_words = ["你", "我", "他"]
df = pd.DataFrame({"words" : [["我", "很", "好"], 
                                   ["你", "是", "誰"],
                                   ["他", "天天", "運動"]]})

df["words"].apply(lambda x : list(filter(lambda x: x not in stop_words, x)))
```

## pandas: find the index of an element in a series
```python
import pandas as pd
myseries = pd.Series([1,4,0,7,5])
print(myseries[myseries==7].index[0]) # 3
```

## pandas: merge two dataframes, on different columns
```python
df1 = pd.DataFrame({"id1" : [1,2,3], "value": [0, 0, 0]})
df2 = pd.DataFrame({"id2" : [1,2,3], "value": [-1, -1, -1]})
print(pd.merge(df1, df2, left_on="id1", right_on="id2"))
```

## pandas: combine columns of text in a dataframe
[Combine two columns of text in dataframe in pandas/python](https://stackoverflow.com/questions/19377969/combine-two-columns-of-text-in-dataframe-in-pandas-python)
```python
df = pd.DataFrame({'Year': ['2014', '2015'], 'quarter': ['q1', 'q2']})
df['period'] = df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
```

## pandas: select rows from dataframe where column_name not in a list
```python
df = pd.DataFrame({'countries':['US','UK','Germany','China']})
countries = ['UK','China']

print(df[~df['countries'].isin(countries)])
"""
  countries
0        US
2   Germany
"""
```

## pandas: where
[Python | Pandas DataFrame.where()](https://www.geeksforgeeks.org/python-pandas-dataframe-where/)
```python
df = df.where((df['col1']>=10) & (df['col2']<=50)).dropna()
```

## pandas: remove duplicated row by looking at a column
```python
df = pd.DataFrame({"name": ["Annie", "Brian", "Cindy", "David", "David"], "score": [90, 85, 76, 86, 87]})
print(df.drop_duplicates(subset="name", keep="last"))
"""
    name  score
0  Annie     90
1  Brian     85
2  Cindy     76
4  David     87
"""
```

## pandas: count elements in a column
```python
df = pd.DataFrame({"name": ["Annie", "Brian", "Cindy", "David", "David"], "score": [90, 85, 76, 86, 87]})
print(df.groupby("name").count())
"""
       score
name        
Annie      1
Brian      1
Cindy      1
David      2
"""
```

## pandas: filter string based on its length
```python
df = pd.DataFrame({"name": ["Ann", "Brian", "Cinderella", "David"], "score": [90, 85, 76, 86]})
mask = (df['name'].str.len() < 5)
print(df.loc[mask])
"""
  name  score
0  Ann     90
"""
```

## pandas: group by a pre-defined bins
[Pandas Groupby Range of Values](https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values)
```python
df = pd.DataFrame({"name": ["Ann", "Brian", "Cinderella", "David", "Eve", "Frank", "Gina", "Helen"], "score": [90, 85, 76, 86, 43, 32, 65, 100]})
print(df.groupby(pd.cut(df["score"], [0, 60, 70, 80, 90, 100])).groups)
"""
{Interval(0, 60, closed='right'): Int64Index([4, 5], dtype='int64'),
 Interval(60, 70, closed='right'): Int64Index([6], dtype='int64'),
 Interval(70, 80, closed='right'): Int64Index([2], dtype='int64'),
 Interval(80, 90, closed='right'): Int64Index([0, 1, 3], dtype='int64'),
 Interval(90, 100, closed='right'): Int64Index([7], dtype='int64')}
"""
```

## pandas: iterate over rows
```python
import pandas as pd

df = pd.DataFrame([{'c1':10, 'c2':100}, {'c1':11,'c2':110}, {'c1':12,'c2':120}])
for index, row in df.iterrows():
    print(row['c1'], row['c2'])

"""
10 100
11 110
12 120
"""
```

## pandas: apply a function on two columns of dataframe
[How to apply a function to two columns of Pandas dataframe](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe?answertab=votes#tab-top)
```python
df['col_3'] = df[['col_1','col_2']].apply(lambda x: f(*x), axis=1)
```

## pandas/matplotlib: plot histogram for a series
```python
fig, ax = plt.subplots()
ax.title.set_text("xxx")
df["col"].hist(ax=ax, bins=100)
ax.set_xscale('log')
ax.set_yscale('log')
fig.show()
fig.savefig('xxx.png')
```

## pandas/matplotlib: plot on secondary y axis
[pandas scatterplots: how to plot data on a secondary y axis?](https://stackoverflow.com/questions/35063104/pandas-scatterplots-how-to-plot-data-on-a-secondary-y-axis)
```python
fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(111)

df.plot(kind='line', x='x', y='y1', label='y1', ax=ax1)
# remove label since it will cover the original one
df.plot(kind='line', x='x', y='y2', style='.', markersize=0, secondary_y=True, ax=ax1, label='_hidden', legend=False)
```

## pandas/matplotlib: remove legend
[How to delete legend in pandas](https://stackoverflow.com/questions/62680533/how-to-delete-legend-in-pandas)
```python
fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(111)

df.plot(kind='line', x='x', y='y', style='.', markersize=0, ax=ax1, label='_hidden', legend=False)
```

## matplotlib: set image size
[How do you change the size of figures drawn with Matplotlib?](https://stackoverflow.com/questions/332289/how-do-you-change-the-size-of-figures-drawn-with-matplotlib)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6), dpi=80)
```

## matplotlib: save image
```python
import matplotlib.pyplot as plt

xs = range(10)
ys = [x*2 for x in xs]
plt.plot(xs, ys)
plt.savefig('xxx.png')
```

## matplotlib: show image in for loop
[Can I generate and show a different image during each loop with Matplotlib?](https://stackoverflow.com/questions/11129731/can-i-generate-and-show-a-different-image-during-each-loop-with-matplotlib)

```python
import cv2
import matplotlib.pyplot as plt

#fimgs = ['a.jpg', 'b.jpg']

for fimg in fimgs:
    img = cv2.imread(fimg)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    _ = input("Press [enter] to continue.") # wait for input from the user
    plt.close()
```

## matplotlib: multiple scatter plots
[MatPlotLib: Multiple datasets on the same scatter plot](https://stackoverflow.com/questions/4270301/matplotlib-multiple-datasets-on-the-same-scatter-plot)
```python
import matplotlib.pyplot as plt

x = range(100)
y = range(100,200)
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
#...
plt.legend(loc='upper left')
plt.show()
```

## matplotlib: plot figure with Chinese characters
[How to plot a figure with Chinese Characters in label](https://stackoverflow.com/questions/39630928/how-to-plot-a-figure-with-chinese-characters-in-label)
```python
from matplotlib import font_manager

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(xs, ys, s=10, label="中文標籤")
plt.title("中文標題", fontproperties=fontP)
plt.xlabel("myxlabel")
plt.ylabel("myylabel")
plt.legend(loc='lower right', prop=fontP)
plt.savefig("myfigure.png")
```

## matplotlib: use tableau colors
```python
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan"
    ]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(xs, ys, c=colors[i], s=10)
```

## matplotlib: put legend right to the plot and save full image
[Plt.show shows full graph but savefig is cropping the image](https://stackoverflow.com/questions/37427362/plt-show-shows-full-graph-but-savefig-is-cropping-the-image)
```python
plt.figure("your_title")
plt.xlabel("your_xlabel")
plt.ylabel("your_ylabel")
plt.plot(xs, ys, '-', label="your_label")
plt.legend(bbox_to_anchor=(1,1)) # put legend right to the plot
plt.savefig("your_title.png", bbox_inches='tight') # otherwise it will save cropped image
```

## matplotlib: draw vertical line
[How to draw vertical lines on a given plot in matplotlib](https://stackoverflow.com/questions/24988448/how-to-draw-vertical-lines-on-a-given-plot-in-matplotlib)
```python
plt.axvline(x=0.1, color='k', linestyle='--')
```

## matplotlib: set x/y axis limit
[setting y-axis limit in matplotlib](https://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib)
```python
ax = plt.gca()
ax.set_xlim([xmin, xmax])
ax.set_ylim([None, ymax]) # None for not setting y axis's lower limit
```

## numpy: float to string with precision
[Python float to string (scientific notation), in specific format](https://stackoverflow.com/questions/59091625/python-float-to-string-scientific-notation-in-specific-format)
```python
np.format_float_scientific(1.23456789, unique=False, exp_digits=2,precision=4)
# '1.2346e+00'
```

## numpy: print without brackets
[How to print a Numpy array without brackets?](https://stackoverflow.com/questions/9360103/how-to-print-a-numpy-array-without-brackets)
```python
np.savetxt(sys.stdout, arr)
```

## numpy: read csv
[How do I read CSV data into a record array in NumPy?](https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy)

[Prevent or dismiss 'empty file' warning in loadtxt](https://stackoverflow.com/questions/19167550/prevent-or-dismiss-empty-file-warning-in-loadtxt)
```python
with warnings.catch_warnings(): # to disable "empty file" warning
    warnings.simplefilter("ignore")
    # the delimiter in a.txt can be ',' or ', '
    cloud = np.genfromtxt(fname, delimiter=' ', dtype=np.float32)
```

## numpy: save csv
[python numpy.savetxt header has extra character #](https://stackoverflow.com/questions/36210977/python-numpy-savetxt-header-has-extra-character/36211002)
```python
np.savetxt(resfname, result, delimiter=',', 
            header="idx1,idx2,t_x,t_y,t_z,q_w,q_x,q_y,q_z",
            fmt='%i,%i,%f,%f,%f,%f,%f,%f,%f')
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

## numpy: count occurrences of unique elements
https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python
```python
a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
unique, counts = np.unique(a, return_counts=True)
print(unique) # array([0, 1, 2, 3, 4])
print(counts) # array([7, 4, 1, 2, 1], dtype=int64)
```

## numpy: convert an array of array(from pandas dataframe) to a 2-D array
```python
df = pd.DataFrame({"vector" : [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9])]})
print(df["vector"].values.shape) #(3,)
print(np.array(list(df["vector"].values)).shape) #(3,3)
```

## numpy: concatenate two 1-D array
```python
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.concatenate([a,b]))
```

## numpy: find index of nonzero elements
```python
arr = np.array([0, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0])
print(np.nonzero(arr)) #(array([3, 4, 8], dtype=int64),)
```

## numpy: remove NaN from an array
```python
import numpy as np

a = np.array([1, np.nan, 3, 4])
print(a[~np.isnan(a)]) # array([1., 3., 4.])
```

## numpy: convert numpy array to list
[NumPy array is not JSON serializable](https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable)
```python
import numpy as np

x = np.zeros((2,3,4))
x = x.tolist()
# [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
# Note that the array's shape doesn't change
```
After it's converted to list, it can be serialized.

## numpy: fill array with a value
[numpy.full](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.full.html)
```python
import numpy as np

x = np.full((2,3), 'a')
# array([['a', 'a', 'a'],
       ['a', 'a', 'a']], dtype='<U1')
```

## numpy: copy an array multiple times
[numpy.tile](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html#numpy.tile)
```python
import numpy as np

# this copy the whole array 3 times
x = np.array([[1,2],[3,4]])
np.tile(x, (3,1))
"""
array([[1, 2],
       [3, 4],
       [1, 2],
       [3, 4],
       [1, 2],
       [3, 4]])
"""

# this copy each row 3 times
np.repeat(x, 3, axis=0)
"""
array([[1, 2],
       [1, 2],
       [1, 2],
       [3, 4],
       [3, 4],
       [3, 4]])
"""
```

## numpy: norm
[numpy.linalg.norm](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html)
```python
from numpy import linalg as LA

x = np.random.rand(10, 128)
LA.norm(x, axis=1)
# array([6.75208915, 6.16545501, 6.20106749, 6.76831967, 6.84324319,
       5.9553995 , 6.48986205, 6.66884094, 6.62251865, 6.59760121])
```

## numpy: multiply two vector and get a matrix
[How to multiply two vector and get a matrix?](https://stackoverflow.com/questions/28578302/how-to-multiply-two-vector-and-get-a-matrix)
```python
numpy.outer(numpy.array([1, 2]), numpy.array([3, 4]))
#array([[3, 4],
#       [6, 8]])
```

## numpy: arange(python "range" supporting float step size
```python
np.arange(0.0, 1.0, 0.1)
```

## numpy: use list as indices
[How to filter numpy array by list of indices?](https://stackoverflow.com/questions/19821425/how-to-filter-numpy-array-by-list-of-indices)
```python
arr = np.arange(15).reshape((3,5))
np.take(arr, [1,3], axis=1)
#array([[ 1,  3],
#       [ 6,  8],
#       [11, 13]])
```

## numpy: add new axis

```python
arr = arr[..., np.newaxis]
```

## numpy: fit line and plot
[How to plot a line of best fit in Python](https://www.kite.com/python/answers/how-to-plot-a-line-of-best-fit-in-python)
```python
x = np.array([1, 3, 5, 7])
y = np.array([ 6, 3, 9, 5 ])
m, b = np.polyfit(x, y, 1) # Adding `cov=True` will also return covariance matrix
plt.plot(x, y, 'o')
plt.plot(x, m*x + b)
plt.show()
```

## skimage: use ransac to find line
[How to fit a line using RANSAC in Cartesian coordinates?](https://stackoverflow.com/questions/59477558/how-to-fit-a-line-using-ransac-in-cartesian-coordinates)

This can be used when outlier ratio is large.

```python
from skimage.measure import LineModelND, ransac

x = np_x.reshape(-1, 1)
y = np_y.reshape(-1, 1)

data = np.column_stack([x, y])

model = LineModelND()
model.estimate(data)
# robustly fit line only using inlier data with RANSAC algorithm
residual_threshold = 0.1 # this should be tuned
model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=residual_threshold,
                               max_trials=1000)

print("inliers: {}/{}".format(np.count_nonzero(inliers), x.size))

origin, direction = model.params
m = direction[1]/direction[0]
b = origin[1] - origin[0] * m
print("m: {}, b: {}".format(m, b))

origin_robust, direction_robust = model_robust.params
m_robust = direction_robust[1]/direction_robust[0]
b_robust = origin_robust[1] - origin[0]_robust * m_robust
print("m_robust: {}, b_robust: {}".format(m_robust, b_robust))
```

## numpy: interpolate a 3D line

[linear interpolation between two data points](https://stackoverflow.com/questions/38282659/linear-interpolation-between-two-data-points)

Given two points (-100, -50, 0) and (0, 50, 100), this script interpolate 10 points between the line connecting them.

```python
x_range = [-100, 0]
y_range = [-50, 50]
z_range = [0, 100]
ranges = [x_range, y_range, z_range]
points = np.zeros((10, 3))
for i in range(0, 10):
    for j in range(3):
        points[i, j] = np.interp(i, [0,10], ranges[j])
```

```
array([[-100.,  -50.,    0.],
       [ -90.,  -40.,   10.],
       [ -80.,  -30.,   20.],
       [ -70.,  -20.,   30.],
       [ -60.,  -10.,   40.],
       [ -50.,    0.,   50.],
       [ -40.,   10.,   60.],
       [ -30.,   20.,   70.],
       [ -20.,   30.,   80.],
       [ -10.,   40.,   90.]])
```

## enumerate all combinations to split an array
[Split an array in all possible combinations (not regular splitting)](https://stackoverflow.com/questions/45780190/split-an-array-in-all-possible-combinations-not-regular-splitting)

```python
from itertools import combinations
import numpy as np

array = [1,2,3,4]
[np.split(array, idx) 
    for n_splits in range(len(array)+1)
    for idx in combinations(range(1, len(array)), n_splits)]

# [[array([1, 2, 3, 4])], [array([1]), array([2, 3, 4])], [array([1, 2]), array([3, 4])], [array([1, 2, 3]), array([4])], [array([1]), array([2]), array([3, 4])], [array([1]), array([2, 3]), array([4])], [array([1, 2]), array([3]), array([4])], [array([1]), array([2]), array([3]), array([4])]]
```

## re: search for text between two strings
[Match text between two strings with regular expression](https://stackoverflow.com/questions/32680030/match-text-between-two-strings-with-regular-expression)
```python
import re

s = "onedir/123.xml"
# this regular expression search for text between '/' and '.'
# note that '/' and '.' should be escaped by '\'
re.search(r'\/(.*?)\.', s).group(1) #123
```

```python
import re

s = """    {"Convolution", parseConvolution},
    {"Pooling", parsePooling},
    {"InnerProduct", parseInnerProduct},
    {"ReLU", parseReLU},
    {"Softmax", parseSoftMax},
    {"SoftmaxWithLoss", parseSoftMax},
    {"LRN", parseLRN},
    {"Power", parsePower},
    {"Eltwise", parseEltwise},
    {"Concat", parseConcat},
    {"Deconvolution", parseDeconvolution},
    {"Sigmoid", parseSigmoid},
    {"TanH", parseTanH},
    {"BatchNorm", parseBatchNormalization},
    {"Scale", parseScale},
    {"Crop", parseCrop},
    {"Reduction", parseReduction},
    {"Reshape", parseReshape},
    {"Permute", parsePermute},
    {"ELU", parseELU},
    {"BNLL", parseBNLL},
    {"Clip", parseClip},
    {"AbsVal", parseAbsVal},
    {"PReLU", parsePReLU}"""

l = re.findall(r'"(.*)"', s)
print(l) # ['Convolution', 'Pooling', 'InnerProduct', 'ReLU', 'Softmax', 'SoftmaxWithLoss', 'LRN', 'Power', 'Eltwise', 'Concat', 'Deconvolution', 'Sigmoid', 'TanH', 'BatchNorm', 'Scale', 'Crop', 'Reduction', 'Reshape', 'Permute', 'ELU', 'BNLL', 'Clip', 'AbsVal', 'PReLU']
```

## re: check if a string is float
[Checking if a string can be converted to float in Python](https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python)
```python
import re

def is_float(s):
    return re.match(r'^-?\d+(?:\.\d+)?$', s) is not None
```

## re: find all digits, including floating points
[if i use re.findall How to register in order not to separate the point](https://stackoverflow.com/questions/44703436/if-i-use-re-findall-how-to-register-in-order-not-to-separate-the-point/44703493)
```python
import re

s = "1: 669.557373, 669.557373 avg, 0.000000 rate, 1.819341 seconds, 256 images"
re.findall("\d+\.\d+|\d+", s) #['1', '669.557373', '669.557373', '0.000000', '1.819341', '256']
```

## re: split with multiple delimeters
[Split string with multiple delimiters in Python [duplicate]](https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python)
```python
import re

s = "0 0.59, 0.73, 0.43, -0.36, 0.00, 0.07, 0.09"
re.split(" |, ", s)
# ['0', '0.59', '0.73', '0.43', '-0.36', '0.00', '0.07', '0.09']
```

## re: remove redundant spaces
[Simple way to remove multiple spaces in a string?](https://stackoverflow.com/questions/1546226/simple-way-to-remove-multiple-spaces-in-a-string)
```python
import re
s = "hello    goodbye hey!"
s = re.sub(' +', ' ', s) # 'hello goodbye hey!'
```

## requests: post data
```python
import requests

response = requests.post("http://<ip-address>:<port>/<subpage>", 
                          data={<key>: <val>})
response = eval(response.text)
```

## requests: save binary file

[How to download image using requests](https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests)

```python
import requests
import shutil

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(fname, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
```

## requests: send image
This can be used with https://github.com/keineahnung2345/cpp-code-snippets/blob/master/crow/imgsave.cpp.
```python
import requests
import base64
import cv2

"""
tried the lines starting with '#' but fail
"""

url = 'http://<ip_address>:<port>/imgsave'
fimg = open('filename.jpg', 'rb')

#headers = {'Content-type': 'application/json; charset=utf-8', 'Accept': 'text/json'}
headers = {'Content-type': 'application/json'}

#https://stackoverflow.com/questions/30280495/python-requests-base64-image
encoded_image = base64.b64encode(fimg.read()).decode('utf-8')
myobj = {'img': encoded_image}

#img = cv2.imread('filename.jpg')
# encode image as jpeg
#_, img_encoded = cv2.imencode('.jpg', img)
#myobj = {'detect_img': str(img_encoded.tostring())}
#myobj = {'detect_img': str(encoded_image)}
#files = {'detect_img': ('filename.jpg', fimg, 'image/jpg', {})}

#x = requests.post(url)
#x = requests.post(url, data = myobj)
#x = requests.post(url, data = json.dumps(myobj))
#x = requests.post(url, data = myobj, headers=headers)
#x = requests.post(url, files = myobj)
#x= requests.post(url, json = myobj, headers = headers)
#x = requests.post(url, files = files, data = {'a':1})
x= requests.post(url, json = myobj)

print(x.text)
```

## glob: to search recursively across sub-folders
[How can I search sub-folders using glob.glob module?](https://stackoverflow.com/questions/14798220/how-can-i-search-sub-folders-using-glob-glob-module)
```python
from glob import glob

glob("/base_dir/**/*.txt", recursive=True)
```

## glob: search only second level sub-folders
[Is there a one-liner to list a directory two levels deep where the second level is an only-child, but not known?](https://stackoverflow.com/questions/4821331/is-there-a-one-liner-to-list-a-directory-two-levels-deep-where-the-second-level)
```python
from glob import glob

glob("base_dir" + "\*\*.txt")
```

## glob: get hidden files
[Python 3.6 glob include hidden files and folders](https://stackoverflow.com/questions/49047402/python-3-6-glob-include-hidden-files-and-folders)
```python
xmls = glob("tmp/**/*.xml", recursive=True)
hidden_xmls = glob("tmp/**/.*xml", recursive=True)
```

## use tqdm with enumerate
```python
l = ['a', 'b', 'c']
for i, e in enumerate(tqdm(l)):
    print(i, e)
```

## use tqdm with while loop
[Using tqdm progress bar in a while loop](https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop)
```python
import random
import numpy as np
from tqdm import tqdm

l = list(range(10000))
pbar = tqdm(total = len(l)) #here!

while len(l) > 0:
    nsample = np.random.choice([1,2,3], 1, p=[0.45,0.45,0.1])[0]
    nsample = min(nsample, len(l))
    sampleds = random.sample(l, nsample)
    for sampled in sampleds:
        l.remove(sampled)
    pbar.update(nsample) #here!

pbar.close() #here!
```

## bisect
find the index of the element just greater than x
```python
import bisect

l = [0, 1]
bisect.bisect_right(l, 0)   #1
bisect.bisect_right(l, 0.3) #1
bisect.bisect_right(l, 1)   #2
```

## opencv: capture image from camera and then close camera
[Capturing a single image from my webcam in Java or Python](https://stackoverflow.com/questions/11094481/capturing-a-single-image-from-my-webcam-in-java-or-python)

[Read, Write and Display a video using OpenCV ( C++/ Python )](https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/)
```python
from cv2 import VideoCapture

# initialize the camera
cap = VideoCapture(0)   # 0 -> index of camera

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Esc on keyboard to exit
        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
```

## opencv: extract width, height, fps, fourcc from a video
Ref: [What is the opposite of cv2.VideoWriter_fourcc?](https://stackoverflow.com/questions/49138457/what-is-the-opposite-of-cv2-videowriter-fourcc)

[Get video dimension in python-opencv](https://stackoverflow.com/questions/39953263/get-video-dimension-in-python-opencv/39953739)

```python
import cv2

def decode_fourcc(cc):
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

cap = cv2.VideoCapture("test_video.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT) # 3, 4
print('width, height: {}, {}'.format(width, height))

fps = round(cap.get(cv2.CAP_PROP_FPS))
print('fps: ', fps)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frame count: ', frame_count)

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
print('fourcc: ', fourcc)

decoded_fourcc = decode_fourcc(fourcc)
print('decoded fourcc: ', decoded_fourcc)

encoded_fourcc = cv2.VideoWriter_fourcc(*decoded_fourcc)
print('encoded_fourcc: ', encoded_fourcc)
```
Sample output:
```
width, height: 1280, 720
fps:  30
frame count:  201
fourcc:  828601953
decoded fourcc:  avc1
encoded_fourcc:  828601953
```

## opencv: from video into images
[Python - Extracting and Saving Video Frames](https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames)

```python
import cv2
import os

video_name = "xxx"
if not os.path.isdir(video_name):
    os.makedirs(video_name)

vidcap = cv2.VideoCapture('{}.mp4'.format(video_name))

success, image = vidcap.read()
count = 0

while success:
    cv2.imwrite("{}/frame{}.jpg".format(video_name, count), image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
```

## opencv: from images into video(.avi and .mp4)
[Writing an mp4 video using python opencv](https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv)
```python
import cv2
import os
import glob

fps = 20
width, height = (1280,720)

extension = ".mp4"
if extension.endswith("avi"):
    # for Mac
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # for linux
    # fourcc = cv2.VideoWriter_fourcc(0x00000021)
elif extension.endswith("mp4"):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter=cv2.VideoWriter("xxx.avi", fourcc, fps, (width, height))
 
imagedir = "/xxx"

for filename in glob.glob(imagedir+"/*.jpg"):
    frame = cv2.imread(filename)
    print(filename)
    videoWriter.write(frame)

videoWriter.release()
```

## opencv: draw rectangle
[Python 與 OpenCV 加入線條圖案與文字教學](https://blog.gtwang.org/programming/opencv-drawing-functions-tutorial/)
```python
import cv2

img = cv2.imread("xxx.jpg")
cv2.rectangle(img, (20, 60), (120, 160), (0, 255, 0), 2)
# cv2.waitKey(0): pause infinitely
cv2.imshow("image", img); cv2.waitKey(0); cv2.destroyAllWindows()
```

## opencv: crop image
[How to crop an image in OpenCV using Python](https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)
```python
import cv2

img = cv2.imread("xxx.jpg")
x, y, h, w = 100, 100, 100, 100
crop_img = img[y:y+h, x:x+w].copy()
cv2.imshow("cropped", crop_img); cv2.waitKey(0); cv2.destroyAllWindows()
```

## opencv: resize image
```python
import numpy as np
import cv2

arr = np.arange(10, dtype='uint8').reshape(2,5,1)
print(arr.shape) # (2, 5, 1)
arr_resized = cv2.resize(arr, (2,5))
print(arr_resized.shape) # (5, 2)
```
The reason for 'uint8': [\resize.cpp:3787: error: (-215:Assertion failed) func != 0 in function 'cv::hal::resize'](https://stackoverflow.com/questions/55087860/resize-cpp3787-error-215assertion-failed-func-0-in-function-cvhal)

Notice the dimension order!
```
(numpy array).shape: (h, w, c)
cv2.resize((numpy array), (w, h))
```

## get image size without loading image
[Get Image size WITHOUT loading image into memory](https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory)
```python
import magic

t = magic.from_file('tmp.jpg')
t
# 'JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 200x130, frames 3'
re.findall('(\d+)x(\d+)', t)
# [('1', '1'), ('200', '130')]
re.findall('(\d+)x(\d+)', t)[-1]
# ('200', '130')
tuple(map(int, re.findall('(\d+)x(\d+)', t)[-1]))
# (200, 130)
```

## convert string to datetime
```python
from datetime import datetime

s_time = "17:18:04"
dt_time = datetime.strptime(s, "%H:%M:%S")
print(dt_time) #1900-01-01 17:18:04
```

## convert datetime to string
[Python strftime()](https://www.programiz.com/python-programming/datetime/strftime)
```python
from datetime import datetime
s_time = datetime.now().strftime("%Y%m%d-%H%M%S")
print(s_time) #20211109-171937
```

## sum up list of timedelta
[How to get the sum of timedelta in Python?](https://stackoverflow.com/questions/4049825/how-to-get-the-sum-of-timedelta-in-python)
```python
from datetime import timedelta

deltas = [timedelta(seconds=208), timedelta(seconds=123), timedelta(seconds=28)]
sum(deltas, timedelta()) #datetime.timedelta(seconds=359)
```

## convert chinese words to pinyin
First install `pypinyin`:
```sh
pip install pypinyin
```
Then:
```python
from pypinyin import lazy_pinyin
print(''.join(lazy_pinyin("an-79种-已标")))
# 'an-79zhong-yibiao'
```

## open file explorer
[Open explorer on a file](https://stackoverflow.com/questions/281888/open-explorer-on-a-file)
```python
import subprocess
subprocess.Popen(r'explorer /select,"C:\"')
```

## run command
[how to run an exe file with the arguments using python](https://stackoverflow.com/questions/15928956/how-to-run-an-exe-file-with-the-arguments-using-python)
```python
# FNULL = open(os.devnull, 'w')    #use this if you want to suppress output to stdout from the subprocess
# subprocess.call(command, stdout=FNULL, stderr=FNULL, shell=False)
# command = exe_path + " -i 1 -s " + '"' + my_str_arg + '"'
subprocess.call(command)
```

## copy file
```python
from shutil import copy

# dst can be a filename or directory name
copy(src, dst)
```

## move/rename file
[How to move a file in Python](https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python)

[Force Overwrite in Os.Rename](https://stackoverflow.com/questions/8107352/force-overwrite-in-os-rename)
```python
import os
import shutil

os.rename("/src/file", "/dst/file") # this will fail if the destination file exist
shutil.move("/src/file", "/dst/file") # this will overwrite the destination file if it exist
```

## remove file
```python
import os

os.remove("/dirname/filename")
```

## copy directory
[How do I copy an entire directory of files into an existing directory using Python?](https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth)
```python
from distutils.dir_util import copy_tree
copy_tree("/src/dir", "/dst/dir")
```

## remove directory
[How do I remove/delete a folder that is not empty with Python?](https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python)
```python
import shutil

shutil.rmtree('/directory_name')
```

## to exit a program
```python
import sys
sys.exit()
```

## solution to `cannot open shared object file`
[Why can't Python find shared objects that are in directories in sys.path?](https://stackoverflow.com/questions/1099981/why-cant-python-find-shared-objects-that-are-in-directories-in-sys-path)
```
Traceback (most recent call last):
  File "darknet.py", line 51, in <module>
    lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
  File "/usr/local/lib/python3.7/ctypes/__init__.py", line 356, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libcudart.so.8.0: cannot open shared object file: No such file or directory
```

Solution: add `LD_LIBRARY_PATH` before `python3`:
```sh
LD_LIBRARY_PATH=/usr/local/lib:`pwd` python3 darknet.py
```
