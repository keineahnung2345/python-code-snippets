# package-installation
Commands to install python packages

## Install pip
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

## Upgrade pip
```bash
# python -m pip install -U pip # Windows
# pip install -U pip # Linux
```

## pip search
pip search: search for PyPI packages whose name or summary contains
```bash
pip search <package-name>
```

## pip, get package's available versions
[Python and pip, list all versions of a package that's available?](https://stackoverflow.com/questions/4888027/python-and-pip-list-all-versions-of-a-package-thats-available)
```bash
pip install <package-name>==
```

## opencv
```bash
apt update -y && apt install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
```
