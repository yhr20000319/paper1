# cora
python main.py --method erm --dataset cora --gnn_gen gcn --gnn gcn --run 20 --lr 0.001 --device 0
python main.py --method erm --dataset cora --gnn_gen gcn --gnn sage --run 20 --lr 0.001 --device 0
python main.py --method erm --dataset cora --gnn_gen gcn --gnn gat --run 20 --lr 0.001 --device 0
python main.py --method erm --dataset cora --gnn_gen gcn --gnn gpr --run 20 --lr 0.001 --device 0

python main.py --method iene --dataset cora --gnn_gen gcn --gnn gcn --lr 0.005 --num_sample 1 --beta 1.0 --lr_a 0.001 --run 20 --device 0 --weight_decay 5e-5 --dropout 0.2
python main.py --method iene --dataset cora --gnn_gen gat --gnn gcn --lr 0.005 --num_sample 1 --beta 1.0 --lr_a 0.001 --run 20 --device 0 --weight_decay 5e-5 --dropout 0.2

# amazon-photo
python main.py --method erm --dataset amazon-photo --gnn_gen gcn --gnn gcn --run 20 --lr 0.001 --device 0
python main.py --method erm --dataset amazon-photo --gnn_gen gat --gnn gcn --run 20 --lr 0.001 --device 0
python main.py --method iene --dataset amazon-photo --gnn_gen gcn --gnn gcn --lr 0.01 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 20 --device 0 --weight_decay 5e-5 --dropout 0.2
python main.py --method iene --dataset amazon-photo --gnn_gen gat --gnn gcn --lr 0.01 --num_sample 1 --beta 1.0 --lr_a 0.005 --run 20 --device 0 --weight_decay 5e-5 --dropout 0.2

