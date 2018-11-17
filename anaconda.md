# anaconda-scripts
Some useful anaconda scripts

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
