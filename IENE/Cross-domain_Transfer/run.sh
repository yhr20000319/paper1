
# Twitch-e
python main.py --dataset twitch-e --method erm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 0
python main.py --dataset twitch-e --method erm --gnn gat --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 0
python main.py --dataset twitch-e --method iene --gnn gcn --lr 0.01 --weight_decay 5e-5 --num_layers 2 --num_sample 5 --beta 3.0 --lr_a 0.001 --device 0 --dropout 0.2
python main.py --dataset twitch-e --method iene --gnn gat --lr 0.01 --weight_decay 5e-5 --num_layers 2 --num_sample 5 --beta 1.0 --lr_a 0.005 --device 0 --dropout 0.2

# fb-100
python main.py --dataset fb100 --method erm --gnn gcn --lr 0.01 --weight_decay 1e-3 --num_layers 2 --device 0
python main.py --dataset fb100 --method iene --gnn gcn --lr 0.01 --weight_decay 5e-5 --num_layers 2 --num_sample 5 --beta 1.0 --lr_a 0.005 --device 0 --dropout 0.2
