# python-code-snippets
Some useful python code snippets

## detect python version inside python kernel
```python
import sys
sys.version_info
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

## get directory name from file name
[Extract a part of the filepath (a directory) in Python](https://stackoverflow.com/questions/10149263/extract-a-part-of-the-filepath-a-directory-in-python)
```python
import os

fname = "/xxx/yyy/zzz/log.txt"
os.path.dirname(fname) #'/xxx/yyy/zzz'
```

## mkdir -p
```python
import os
# this works even when 'first_layer' doesn't exist
os.makedirs("first_layer/second_layer")
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

## argparse: add positional argument and optional argument, argument with default value
[Python argparse: default value or specified value](https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value)
```python
import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument("pos1")
parser.add_argument("-o1", "--optional1", dest="o1")
parser.add_argument("-i1", dest="i1", type=int, default=0)
args = parser.parse_args()
pos1 = args.pos1
o1 = args.o1
i1 = args.i1
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

## re: search for text between two strings
[Match text between two strings with regular expression](https://stackoverflow.com/questions/32680030/match-text-between-two-strings-with-regular-expression)
```python
import re

s = "onedir/123.xml"
# this regular expression search for text between '/' and '.'
# note that '/' and '.' should be escaped by '\'
re.search(r'\/(.*?)\.', s).group(1) #123
```

## re: find all digits, including floating points
[if i use re.findall How to register in order not to separate the point](https://stackoverflow.com/questions/44703436/if-i-use-re-findall-how-to-register-in-order-not-to-separate-the-point/44703493)
```python
import re

s = "1: 669.557373, 669.557373 avg, 0.000000 rate, 1.819341 seconds, 256 images"
re.findall("\d+\.\d+|\d+", s) #['1', '669.557373', '669.557373', '0.000000', '1.819341', '256']
```

## requests: post data
```python
import requests

response = requests.post("http://<ip-address>:<port>/<subpage>", 
                          data={<key>: <val>})
response = eval(response.text)
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

## opencv: capture image from camera and then close camera
[Capturing a single image from my webcam in Java or Python](https://stackoverflow.com/questions/11094481/capturing-a-single-image-from-my-webcam-in-java-or-python)
```python
from cv2 import VideoCapture

# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
res, img = cam.read()
cam.release() # close camera
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

## opencv: from images into video(.avi)
```python
import cv2
import os
import glob

fps = 20
width, height = (1280,720)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
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

## convert string to datetime
```python
from datetime import datetime

s_time = "17:18:04"
dt_time = datetime.strptime(s, "%H:%M:%S")
print(dt_time) #1900-01-01 17:18:04
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

## copy file
```python
from shutil import copy

# dst can be a filename or directory name
copy(src, dst)
```

## move file
[How to move a file in Python](https://stackoverflow.com/questions/8858008/how-to-move-a-file-in-python)
```python
import os
import shutil

os.rename("/src/file", "/dst/file")
shutil.move("/src/file", "/dst/file")
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
