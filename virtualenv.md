# virtualenv

## Install virtualenv
```bash
pip3 install -U pip virtualenv
```

## Create a virtual environment
This will create a virtual environment named `virtual_environment_name` in `./virtual_environment_name` directory. 
```bash
virtualenv --system-site-packages -p python3 ./virtual_environment_name
```

## Activate a virtual environment
```bash
.\virtual_environment_name\Scripts\activate
```

## Deactivate
```bash
deactivate
```

## Remove a virtual environment
From [How do I remove/delete a virtualenv?](https://stackoverflow.com/questions/11005457/how-do-i-remove-delete-a-virtualenv):
```bash
# simply delete the folder named ./virtual_environment_name
```
