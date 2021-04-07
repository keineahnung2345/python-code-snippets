"""
This script normalizes the titles of wma files:
トラック 1 -> トラック 01
        1 ->         01
  Track 1 ->   Track 01
"""

import taglib
from glob import glob
import re

song_dirs = [
    "wma_dir"
    ]

for song_dir in song_dirs:
    for songf in glob(song_dir + "/*.wma"):
        try:
            song = taglib.File(songf)
            title = song.tags['TITLE'][0]
            if re.match("^Track \d$", title):
                #https://stackoverflow.com/questions/2763750/how-to-replace-only-part-of-the-match-with-python-re-sub
                song.tags['TITLE'] = [re.sub(r'(\_a)? ([^\.]*)$' , r' 0\2', title)]
                song.save()
            elif re.match("^\d$", title):
                song.tags['TITLE'] = ["0" + title]
                song.save()
            elif re.match("^トラック \d$", title):
                song.tags['TITLE'] = [re.sub(r'(\_a)? ([^\.]*)$' , r' 0\2', title)]
                song.save()
            song.close()
        except:
            print("cannot read", songf) 

