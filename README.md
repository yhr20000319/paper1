## The code of IENE 
WWW 2025 anonymous submission. Number: 1268.

## Download Datasets
We used the datasets provided by [Wu et al.](https://github.com/qitianwu/GraphOOD-EERM). We slightly modified their code to support data loading. 

You can make a directory `./IENE/data` and download all the datasets through the Google drive:
```
https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing
```
Make sure the data files are in the `./IENE/data` folder:
```
README.md
IENE
│
└───Artificial_Transformation
└───Cross-domain_Transfer
└───Temporal_Evolution
└───data   
│   Amazon
│   elliptic
│   ...
```
## Dependency
```
PYTHON>=3.7
PyTorch>=1.9.0
PyTorch Geometric>=1.7.2
ogb>=1.3.4
gpytorch>=1.11
dgl>=0.6.1
```
## Running the code
```shell
# cora
python main.py --method erm --dataset cora --gnn_gen gcn --gnn gcn --run 20 --lr 0.001 --device 0
python main.py --method iene --dataset cora --gnn_gen gcn --gnn gcn --lr 0.005 --num_sample 1 --beta 1.0 --lr_a 0.001 --run 20 --device 0 --weight_decay 5e-5 --dropout 0.2
```
