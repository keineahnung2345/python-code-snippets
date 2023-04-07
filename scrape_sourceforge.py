# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:14:59 2020

This script is used to download Point Cloud Library Files.
"""

import requests
from bs4 import BeautifulSoup
import wget #for downloading files
# convert url to normal string
from urllib.parse import unquote
import os

url_base = "https://sourceforge.net/"
url_start = "https://sourceforge.net/projects/pointclouds/files/PCD%20datasets/"

def download_folder(url):
    if url[-1] == '/':
        url = url[:-1]
    
    folder = unquote(url.split('/')[-1])
    if not os.path.exists(folder):
        os.mkdir(folder)
    os.chdir(folder)
    print("go to", folder)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, features="lxml")
    items = soup.findAll("span", attrs={"class" : "name"})
    items = [item.parent.get("href").replace("/download", "") for item in items]
    print(len(items))

    files = []
    nexturls = []
    for item in items:
        if not item.endswith("/"):
            files.append(item)
        else:
            nexturls.append(url_base + item)
            
    for file in files:
        print("download ", file)
        # wget.download(file)
        """
        need to install the latest windows wget from
        https://eternallybored.org/misc/wget/
        """
        os.system("wget --continue " + file)
    
    for nexturl in nexturls:
        download_folder(nexturl)
    
    os.chdir("..")

if __name__ == "__main__":
    download_folder(url_start)
