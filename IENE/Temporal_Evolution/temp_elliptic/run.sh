
# elliptic

python main.py --method erm --gnn sage --lr 0.01 --weight_decay 0. --num_layers 5 --dataset elliptic --device 0

python main.py --method iene --gnn sage --lr 0.001 --weight_decay 0. --num_layers 5 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --dataset elliptic --device 0

python main.py --method erm --gnn gpr --lr 0.01 --weight_decay 0. --num_layers 5 --dataset elliptic --device 0

python main.py --method iene --gnn gpr --lr 0.01 --weight_decay 0. --num_layers 5 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --dataset elliptic --device 0