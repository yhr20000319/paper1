
# ogb-arxiv
python main.py --method erm --gnn gcn --lr 0.01 --dataset ogb-arxiv --device 0
python main.py --method erm --gnn sage --lr 0.01 --dataset ogb-arxiv --device 0
python main.py --method erm --gnn gpr --lr 0.01 --dataset ogb-arxiv --device 0
python main.py --method iene --gnn gcn --lr 0.01 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device 0 --dropout 0.2 --weight_decay 5e-4
python main.py --method iene --gnn sage --lr 0.01 --num_sample 1 --beta 0.5 --lr_a 0.01 --dataset ogb-arxiv --device 0 --dropout 0.2 --weight_decay 5e-4
python main.py --method iene --gnn gpr --lr 0.01 --num_sample 1 --beta 1.0 --lr_a 0.001 --dataset ogb-arxiv --device 0 --dropout 0.2 --weight_decay 5e-4

