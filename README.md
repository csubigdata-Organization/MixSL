## Requirements

```shell
python == 3.7.4
torch == 1.8.1+cu111
torch-cluster == 1.5.9
torch_scatter == 2.0.8  
torch-sparse == 0.6.12
torch-geometric == 2.2.0
```



## Quick Start

A quick start example is as follows:

using graph substructure feature:
```shell
$ python main.py --data_name MUTAG --backbone GIN --use_substructure_feature True
```

using original feature:
```shell
$ python main.py --data_name MUTAG --backbone GIN --use_substructure_feature False
```



## Complete Start

If running other datasets (such as NCI1), please perform complete process:

### First, generate the graph substructure feature:
```shell
$ python get_encoded_feature.py --data_name NCI1
```

### Then, perform mixed substructure learning:

using graph substructure feature:
```shell
$ python main.py --data_name NCI1 --backbone GIN --use_substructure_feature True
```

using original feature:
```shell
$ python main.py --data_name NCI1 --backbone GIN --use_substructure_feature False
```
