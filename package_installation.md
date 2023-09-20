# package-installation
Commands to install python packages

## to accelerate pip install
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package_name>
```

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

## check package version
[How to check version of python modules?](https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules)
```bash
pip freeze | grep <package-name>
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

## matplotlib
```python
python3 -m pip install matplotlib
# to solve ImportError: No module named '_tkinter'
apt-get install python3-tk
```

## opencv
Linux:
```bash
apt update -y && apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0
pip install opencv-python
```

## PIL
[Installing PIL with pip](https://stackoverflow.com/questions/20060096/installing-pil-with-pip)
```python
python3 -m pip install Pillow
```

## magic
[ImportError: failed to find libmagic](https://github.com/tyiannak/pyAudioAnalysis/issues/128)

Linux:
```sh
pip install python-magic
```
Mac:
```sh
pip install python-magic-bin
```

## glob
[How to install the 'glob' module?](https://stackoverflow.com/questions/42964691/how-to-install-the-glob-module)
```sh
pip install glob3
```

## gpustat
[wookayin/gpustat](https://github.com/wookayin/gpustat)
```sh
pip install gpustat
```

## python-ldap
[python-ldap](https://www.python-ldap.org/en/python-ldap-3.4.3/)

[ERROR: Could not build wheels for python-ldap, which is required to install pyproject.toml-based projects](https://stackoverflow.com/questions/75736939/error-could-not-build-wheels-for-python-ldap-which-is-required-to-install-pypr)
```sh
sudo apt install python3-dev libxml2-dev libxslt1-dev zlib1g-dev libsasl2-dev libldap2-dev build-essential libssl-dev libffi-dev libmysqlclient-dev libjpeg-dev libpq-dev libjpeg8-dev liblcms2-dev libblas-dev libatlas-base-dev
pip install python-ldap
```

## yaml
[How do I install the yaml package for Python?](https://stackoverflow.com/questions/14261614/how-do-i-install-the-yaml-package-for-python)
```sh
pip install pyyaml
```
