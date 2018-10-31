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
