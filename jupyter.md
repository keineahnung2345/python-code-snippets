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
