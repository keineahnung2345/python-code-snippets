import os
import shutil
from glob import glob
import random

# this should be executed at the direcotry containing directories you want to sample
# this script will sample from directories under current directory, and
#  create a destination directory named "mysample" containing the samples

srcdirs = os.listdir(os.getcwd())
srcdirs = [srcdir for srcdir in srcdirs if os.path.isdir(srcdir)]
print(len(srcdirs))

dstdir = "mysample"
sample_count = 5
if dstdir in srcdirs:
    srcdirs.remove(dstdir)
    shutil.rmtree(dstdir)
os.mkdir(dstdir)

for srcdir in srcdirs:
    imgs = glob(srcdir + "/*.jpg")
    random.shuffle(imgs)
    dst_class_dir = os.path.join(dstdir, srcdir)
    os.mkdir(dst_class_dir)
    for img in imgs[:sample_count]:
        shutil.copy(img, dst_class_dir)
