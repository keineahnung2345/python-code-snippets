# -*- coding: utf-8 -*-
# https://www.it145.com/9/126045.html
# or using Live Stream Downloader
# https://chrome.google.com/webstore/detail/live-stream-downloader/looepbdllpjgdmkpdcdffhdbmpbcfekj
import requests
import os
from pathlib import Path
import wget
import subprocess
from tqdm import tqdm

def downloadUrl(url, save_path):
    # https://yanwei-liu.medium.com/%E7%94%A8python%E4%B8%8B%E8%BC%89%E6%AA%94%E6%A1%88-451d1b6f5c10
    wget.download(url, save_path)

def getTsUrl(m3u8Fname, baseUrl):
    if not baseUrl.endswith("/"):
        baseUrl += "/"
    ts_url_list = set()
    with open(m3u8Fname, "r", encoding="utf-8") as f:
        m3u8Contents = f.readlines()
        m3u8Contents = [line.strip() for line in m3u8Contents]
        for content in m3u8Contents:
            if ".ts" in content:
                content = content.split(".ts")[0].replace("..", "")+".ts"
                ts_Url = baseUrl + content
                ts_url_list.add(ts_Url)
    return list(ts_url_list)

def download_ts_video(download_path, ts_url_list):
    if not download_path.endswith("/"):
        download_path += "/"
    for i in tqdm(range(len(ts_url_list))):
        ts_url = ts_url_list[i]
        try:
            response = requests.get(ts_url, stream=True, verify=False)
        except Exception as e:
            print("request exception：%s" % e.args)
            return
        ts_path = download_path + "{}.ts".format(i)
        with open(ts_path, "wb+") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    print("Ts files download complete!")

def mergeTsVideo(download_path,merge_path):
    merge_ts_path = merge_path.rsplit(".",1)[0] + ".ts"
    merge_mp4_path = merge_path.rsplit(".",1)[0] + ".ts"
    # all_ts = os.listdir(download_path)
    # with open(merge_ts_path, 'wb+') as f:
    #     for i in tqdm(range(len(all_ts))):
    #         ts_video_path = os.path.join(download_path, all_ts[i])
    #         f.write(open(ts_video_path, 'rb').read())
    
    # https://stackoverflow.com/questions/37105973/converting-ts-to-mp4-with-python
    subprocess.run(['ffmpeg', '-i', merge_ts_path, merge_mp4_path])
    print("Merge complete!")

if __name__ == '__main__':
    # https://stackoverflow.com/questions/4028904/what-is-the-correct-cross-platform-way-to-get-the-home-directory-in-python
    base_download_path = os.path.join(str(Path.home()), "Downloads", "cvlife")
    m3u8Routes = [
        "https://xxx.com/yyy/zzz.m3u8"
        ]
    
    for m3u8Route in m3u8Routes:
        baseUrl, m3u8Fname = m3u8Route.split("?")[0].rsplit("/", 1)
        download_dir = os.path.join(base_download_path, m3u8Fname.rsplit(".", 1)[0])
        os.makedirs(download_dir, exist_ok=True)
        m3u8Path = os.path.join(download_dir, m3u8Fname)
        downloadUrl(m3u8Route, m3u8Path)
        ts_url_list = getTsUrl(m3u8Path, baseUrl)
        download_ts_video(download_dir, ts_url_list)
        merge_path = os.path.join(download_dir, "xxx.mp4")
        mergeTsVideo(download_dir,merge_path)
