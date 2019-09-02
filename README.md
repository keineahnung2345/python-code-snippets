# python-code-snippets
Some useful python code snippets

## detect python version inside python kernel
```python
import sys
sys.version_info
```

## check file size
```python
import os
statinfo = os.stat('xxx.txt')
print(statinfo) # os.stat_result(st_mode=33188, st_ino=6911355, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=26571, st_atime=1565929634, st_mtime=1565346343, st_ctime=1565929634)
print(statinfo.st_size) # 26571 # the unit is byte
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
with open('stop_words.txt', 'r', encoding='utf-8-sig') as f:
    print(f.readlines())
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

## count the occurence of elements in a list
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

## convert all values of a dict to 1
```python
d = {0:1, 1:4, 2:7, 3:1, 4:9, 5:2}

print(dict(zip(d.keys(), [1]*len(d))))
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

## requests: post data
```python
import requests

response = requests.post("http://<ip-address>:<port>/<subpage>", 
                          data={<key>: <val>})
response = eval(response.text)
```

## glob: to search recursively across sub-folders
[How can I search sub-folders using glob.glob module?](https://stackoverflow.com/questions/14798220/how-can-i-search-sub-folders-using-glob-glob-module)
```python
from glob import glob

glob("/base_dir/**/*.txt", recursive=True)
```

## use tqdm with enumerate
```python
l = ['a', 'b', 'c']
for i, e in enumerate(tqdm(l)):
    print(i, e)
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

## copy file
```sh
from shutil import copy

# dst can be a filename or directory name
copy(src, dst)
```

## remove directory
[How do I remove/delete a folder that is not empty with Python?](https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python)
```sh
import shutil

shutil.rmtree('/directory_name')
```

## to exit a program
```sh
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
