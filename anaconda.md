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
