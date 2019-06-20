# jupyter commands
Some useful commands when doning with jupyter

## Set passowrd to your jupyter notebook instead of using a token
```sh
jupyter notebook password
```

## Start jupyter notebook at specific port
Note： --no-browser for working in command-line interface
```sh
jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=<your-port>
```

## Check running jupyter notebooks, you can check which ports are used here
```sh
jupyter notebook list
```

## Stop running jupyter notebook using their port
```sh
jupyter notebook stop <your-port>
```

## Use tqdm in jupyter notebook
```py
from tqdm import tqdm_notebook as tqdm
```

## Use matplotlib in jupyter notebook
```py
%matplotlib notebook
```

## Jupyter notebook spell checking
[Spell checking in Jupyter notebook markdown cells](http://qingkaikong.blogspot.com/2018/09/spell-checking-in-jupyter-notebook.html)
```sh
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable spellchecker/main
```

## Show differences between jupyter notebooks
[jupyter/nbdime](https://github.com/jupyter/nbdime)
```sh
pip install nbdime
nbdiff a.ipynb b.ipynb
```

## Solve ImportError: cannot import name 'create_prompt_application'
[ipython - "cannot import name 'create_prompt_application' from 'prompt_toolkit.shortcuts'](https://stackoverflow.com/questions/51676835/ipython-cannot-import-name-create-prompt-application-from-prompt-toolkit-s)
```sh
pip install 'prompt-toolkit<2.0.0,>=1.0.15' --force-reinstall
```
Traceback:
```
Traceback (most recent call last):
  File "/opt/conda/lib/python3.6/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/opt/conda/lib/python3.6/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py", line 15, in <module>
    from ipykernel import kernelapp as app
  File "/opt/conda/lib/python3.6/site-packages/ipykernel/__init__.py", line 2, in <module>
    from .connect import *
  File "/opt/conda/lib/python3.6/site-packages/ipykernel/connect.py", line 13, in <module>
    from IPython.core.profiledir import ProfileDir
  File "/opt/conda/lib/python3.6/site-packages/IPython/__init__.py", line 55, in <module>
    from .terminal.embed import embed
  File "/opt/conda/lib/python3.6/site-packages/IPython/terminal/embed.py", line 16, in <module>
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
  File "/opt/conda/lib/python3.6/site-packages/IPython/terminal/interactiveshell.py", line 22, in <module>
    from prompt_toolkit.shortcuts import create_prompt_application, create_eventloop, create_prompt_layout, create_output
ImportError: cannot import name 'create_prompt_application'
```
