# anaconda-scripts
Some useful anaconda scripts

## list all environments
```sh
conda env list
```

## create an environment
```sh
conda create -n <env-name> python=<python-version>
```

## remove an environment
```sh
conda env remove -n <env-name>
```

## show the config of anaconda
```sh
conda config --show
```

## set a parameter in anaconda config file
```sh
conda config --set <parameter-name> <value>
```

For example:
```sh
conda config --set remote_connect_timeout_secs 60.0
```

## conda install too slow
[solving environment for 6 hours](https://github.com/conda/conda/issues/7690#issuecomment-451582942)
```sh
conda install --override-channels -c main -c conda-forge <package-name>
```
