## Data:

Dataloaders load data from datasets that are in yolo format. To convert other (or custom) datasets into the yolo format, you can create your own converter script that takes in a source path and a destination path. In the destination path, the file structure is the following:

```
root/
|---- images
     |---- {id}.jpeg
     |---- ...
|---- labels
     |---- train
          |---- {id}.txt
          |---- ...
     |---- val
          |---- {id}.txt
          |---- ...
```

## Troubleshooting:

#### ModuleNotFoundError:
If running a script (for exapmle the pascalvoc_to_yolo conversion script) like so:

```
python -m data.pascalvoc.pascalvoc_to_yolo
```

first append the project's root path to `PYTHONPATH`, which you can do on linux with the following command (run the command from the project's root dir):

```
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```
