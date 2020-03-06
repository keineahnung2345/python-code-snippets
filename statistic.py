"""
This script is used to get the count of all classes from xml annotations files
Usage:
python3 statistic.py --labeldir ~/VOCdevkit/VOC2019/Annotations
python3 statistic.py --labelfile ~/VOCdevkit/VOC2019/train.txt
python3 statistic.py --metafile ~/VOCdevkit/VOC2019/train.txt
"""
from collections import Counter
import sys
import os
import time
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import uuid

correct_name_map = {}

with open('class_correction.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        correct_name_map[line.split()[0]] = line.split()[1]

def correct_name(name):
    name = name.strip().replace(' ', '').replace('.', '')
    if name in correct_name_map:
        return correct_name_map[name]
    return name

def get_filename(filepath, filetype):

    filename = []
    for root, dirs, files in os.walk(filepath):
        for i in files:
            #if filetype in i:       
            if i.endswith(filetype):       
                fileidx = root + '/' + i            
                if os.path.exists(fileidx):
                    filename.append(fileidx)
                else:
                    print(fileidx)
    return filename

def statics_classes_from_labels(src_dir, num_classes):
    
    filetype = '.txt'
    filenames = get_filename(src_dir, filetype)
    hist = list()

    for i in range(len(filenames)):
        #label_name = src_dir + '/' + filenames[i]
        #print(filenames[i])
        label_name = filenames[i]
        with  open(label_name , 'r') as fid:
            lines = fid.readlines()

            for j,line in enumerate(lines):
                line = line.split(' ')
                class_label = line[0]
                hist.append(class_label)

    print("total labels is: ", len(hist))
    result = Counter(hist)
    for k in range(0, num_classes):

        print(str(k)+"  count is:"+str(result[str(k)]))

    print("total class is: ", len(result))

    print(result)

def statics_classes_from_xml_dir(src_dir):
    
    filetype = '.xml'
    filenames = get_filename(src_dir, filetype)
    return statics_classes_from_xml_filenames(filenames)

def statics_classes_from_metafile(metafile):
    with open(metafile, 'r') as f:
        filenames = f.readlines()
        filenames = [filename.strip() for filename in filenames]
        filenames = [filename.replace('JPEGImages', 'Annotations').replace('jpg','xml') for filename in filenames]
    return statics_classes_from_xml_filenames(filenames)

def statics_classes_from_xml_filenames(filenames):
    label_dict = dict()
    emptyfiles = []
    for i in tqdm(range(len(filenames))):
        #xml_file = src_dir + '/' + filenames[i]
        xml_file = filenames[i]
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                name = obj.find('name').text
                name = correct_name(name)
                obj.find('name').text = name
                if name in label_dict:
                    label_dict[name] = label_dict[name] + 1
                else:
                    label_dict[name] = 1
            tree.write(xml_file)
        except:
            emptyfiles.append(xml_file)
            #print("empty xml!!!", xml_file)

    print("all label: ", len(label_dict))
    for key, value in sorted(label_dict.items()):
        print(key, value)
    with open(os.path.join(os.getcwd(), "statistic_{}.pkl".format(str(uuid.uuid4()))), "wb") as f:
        pickle.dump(label_dict, f)
    return emptyfiles

def statics_classes_from_labelfile(labelfile):
    label_dict = dict()
    with open(labelfile, 'r') as f:
        files = f.readlines()
        files = [f.strip().replace('JPEGImages','labels').replace('.jpg', '.txt') for f in files]
    
    for fname in tqdm(files):
        with open(fname, 'r') as f:
            lines = f.readlines()
            labels = [int(l.split()[0]) for l in lines]
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = 1
                else:
                    label_dict[label] += 1
    print("all label: ", len(label_dict))
    for key, value in sorted(label_dict.items()):
        print(key, value)
    with open(os.path.join(os.getcwd(), "statistic_{}.pkl".format(str(uuid.uuid4()))), "wb") as f:
        pickle.dump(label_dict, f)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--labeldir", dest="labeldir")
    parser.add_argument("--metafile", dest="metafile")
    parser.add_argument("--labelfile", dest="labelfile")
    args = parser.parse_args()
    start = time.time()
    if args.labeldir is not None:
        labeldir = args.labeldir
        #num_classes = sys.argv[2]
        print(labeldir)
        #print(num_classes)
        #statics_classes_from_labels(label_dir, int(num_classes))
        emptyfiles = statics_classes_from_xml_dir(labeldir)
        with open('emptyfiles.txt', 'w') as f:
            emptyfiles = [f+'\n' for f in emptyfiles]
            f.writelines(emptyfiles)
        print(len(emptyfiles), "empty files")
    elif args.metafile is not None:
        metafile = args.metafile
        print(metafile)
        emptyfiles = statics_classes_from_metafile(metafile)
        with open('emptyfiles.txt', 'w') as f:
            emptyfiles = [f+'\n' for f in emptyfiles]
            f.writelines(emptyfiles)
        print(len(emptyfiles), "empty files")
    elif args.labelfile is not None:
        labelfile = args.labelfile
        print(labelfile)
        statics_classes_from_labelfile(labelfile)
    else:
        print('must pass --labeldir or --metafile or --labelfile!')
    end = time.time()
    print("time is: ", end - start)
