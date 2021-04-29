# -*- coding: utf-8 -*-

"""
This script should be used in Windows, 
it visits multiple dirs containing ts files,
and merges them into mp4 files(one dir for one mp4 file)

Directory structure:
basedir
├─m3u8_file1
├─m3u8_file2
├─m3u8_file3
├─...
├─m3u8_dir1
│  ├─tsfile1
│  ├─tsfile2
│  ├─tsfile3
├─m3u8_dir2
├─m3u8_dir3
├─...
"""

import os
from glob import glob

basedir = "D:/python/"

for m3u8fname in glob("*.m3u8"):
    os.chdir(basedir)
    
    with open(m3u8fname, "r") as f:
        lines = f.readlines()
        tsfnames = [line.strip() for line in lines if "ts" in line]
    
    m3u8dirname = m3u8fname.rsplit('.', 1)[0]
    
    # not scrapped yet
    if not os.path.exists(m3u8dirname):
        continue
    
    # not completely scrapped yet
    if len(glob(m3u8dirname + "/*.ts")) != len(tsfnames):
        continue
    
    # already generate mp4 file
    if os.path.exists(m3u8dirname+".mp4"):
        continue
    
    joined_tsfnames = "+".join(tsfnames).replace('\\', '/')
    print("cd to ", basedir + m3u8dirname)
    os.chdir(basedir + m3u8dirname)
    
    # merge multiple ts files into a mp4 file
    cmd = "copy /b " + joined_tsfnames + "  " + m3u8dirname + ".mp4"
    print(cmd)
    os.system(cmd)
    
    # move the mp4 file to basedir
    cmd = "mv " + m3u8dirname + ".mp4 .."
    print(cmd)
    os.system(cmd)
    
