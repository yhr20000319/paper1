
# Twitch-e
python main.py --dataset twitch-e --method erm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 0

python main.py --dataset twitch-e --method iene --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 0

python main.py --dataset twitch-e --method erm --gnn gat --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 0

python main.py --dataset twitch-e --method iene --gnn gat --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --device 0

python main.py --dataset twitch-e --method erm --gnn gcnii --lr 0.01 --weight_decay 1e-3 --num_layers 10 --device 0

python main.py --dataset twitch-e --method iene --gnn gcnii --lr 0.001 --weight_decay 1e-3 --num_layers 10 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.01 --device 0

# fb-100
python main.py --dataset fb100 --method erm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 0

python main.py --dataset fb100 --method iene --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --K 3 --T 1 --num_sample 5 --beta 1.0 --lr_a 0.005 --device 0