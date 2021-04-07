# -*- coding: utf-8 -*-

"""
this script parses the output of diff -rq <dir1> <dir2>
ref: https://github.com/keineahnung2345/linux-commands#show-difference-between-two-directories
"""

from collections import defaultdict
from tqdm import tqdm

with open("diff_log.txt") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

case2samples = dict()
case2samples["bin"] = []
case2samples["only"] = []
case2samples["diff"] = []
case2samples["other"] = []

in_diff = False

for line in tqdm(lines):
    if line.startswith("diff -r"):
        in_diff = True
        case2samples["diff"].append(line)
    elif line.startswith("Binary files"):
        in_diff = False
        case2samples["bin"].append(line)
    elif line.startswith("Only in"):
        in_diff = False
        case2samples["only"].append(line)
    elif in_diff:
        case2samples["diff"][-1] += "\n" + line
    else:
        case2samples["other"].append(line)
