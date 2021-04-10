# -*- coding: utf-8 -*-

import requests
from tqdm import tqdm
import shutil
import os

m3u8_fname = "index.m3u8"
base_url = "https://www.dt870.com/20190228/Je3nEuse/1000kb/hls"
out_dir = "Video/"

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

with open(m3u8_fname, "r") as f:
    tss = f.readlines()
    tss = [ts.strip() for ts in tss if not ts.startswith('#')]

failed = []

for ts in tqdm(tss):
    response = requests.get(base_url + "/" + ts, stream=True)
    if response.status_code == 200:
        with open(out_dir + "/" + ts, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    else:
        print(ts, "failed!")
        failed.append(ts)

print("There are ", len(failed), " failed ts.")

with open("failed.txt", "a") as f:
    f.writelines(failed)
