import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_whole_graph, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, load_fixed_splits
from parse import parse_method_base, parse_method_ours, parser_add_main_args

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(dataset, sub_dataset=None):
    ### Load and preprocess data ###
    if dataset == 'elliptic':
        dataset = load_nc_dataset(args.data_dir, 'elliptic', sub_dataset)
    else:
        raise ValueError('Invalid dataname')

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'], dataset.graph['node_feat']

    return dataset

if args.dataset == 'elliptic':
    tr_subs, val_subs, te_subs = [i for i in range(6, 11)], [i for i in range(11, 16)], [i for i in range(16, 49)]
    datasets_tr = [get_dataset(dataset='elliptic', sub_dataset=tr_subs[i]) for i in range(len(tr_subs))]
    datasets_val = [get_dataset(dataset='elliptic', sub_dataset=val_subs[i]) for i in range(len(val_subs))]
    datasets_te = [get_dataset(dataset='elliptic', sub_dataset=te_subs[i]) for i in range(len(te_subs))]
else:
    raise ValueError('Invalid dataname')

dataset_tr = datasets_tr[0]
dataset_val = datasets_val[0]
print(f"Train num nodes {dataset_tr.n} | num classes {dataset_tr.c} | num node feats {dataset_tr.d}")
print(f"Val num nodes {dataset_val.n} | num classes {dataset_val.c} | num node feats {dataset_val.d}")
for i in range(len(te_subs)):
    dataset_te = datasets_te[i]
    print(f"Test {i} num nodes {dataset_te.n} | num classes {dataset_te.c} | num node feats {dataset_te.d}")

### Load method ###
if args.method == 'erm':
    model = parse_method_base(args, datasets_tr, device)
else:
    model = parse_method_ours(args, datasets_tr, device)

# using rocauc as the eval function
if args.rocauc or args.dataset in ('twitch-e', 'fb100', 'elliptic'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_f1
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc
CELoss = torch.nn.CrossEntropyLoss()
logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)
print('DATASET:', args.dataset)

### Training loop ###
for run in range(args.runs):
    if args.method == 'erm':
        model = parse_method_base(args, datasets_tr, device)
    else:
        model = parse_method_ours(args, datasets_tr, device)
    if args.method == 'erm':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.method == 'iene':
        optimizer_ir_learner = torch.optim.SGD(list(model.ir_Learner.parameters())
                                               + list(model.decoder.parameters()), lr=args.lr_a)
        optimizer_gnn_cls = torch.optim.AdamW(list(model.gnn.parameters())
                                              + list(model.cls.parameters())
                                              + list(model.e_cls.parameters()), lr=args.lr,
                                              weight_decay=args.weight_decay)
        dif_cls_list = list(model.dif_cls[0].parameters())
        for i in range(1, args.e):
            dif_cls_list = dif_cls_list + list(model.dif_cls[i].parameters())
        optimizer_cls = torch.optim.AdamW(dif_cls_list, lr=args.lr,
                                          weight_decay=args.weight_decay)
        optimizer_env_cls = torch.optim.AdamW(model.e_cls.parameters(), lr=args.lr_a)
        for i, dataset_tr in enumerate(datasets_tr):
            model.init_env_adj(dataset_tr)
    best_val = float('-inf')

    if args.method == 'iene':
        model.train()
        for epoch in range(args.pre_epochs):
            for i, dataset_tr in enumerate(datasets_tr):
                x = dataset_tr.graph['node_feat'].to(args.device)
                edge_index = dataset_tr.graph['edge_index'].to(args.device)
                y = dataset_tr.label.squeeze(-1).to(args.device)
                # minimize dif_cls
                Mean = model(dataset_tr, criterion, step=1, id=i)
                dif_cls_loss = Mean
                optimizer_cls.zero_grad()
                dif_cls_loss.backward()
                optimizer_cls.step()

                # minimize cls and gnn_inv with dif_cls
                Mean = model(dataset_tr, criterion, step=2, id=i)
                cls_loss = Mean
                optimizer_gnn_cls.zero_grad()
                cls_loss.backward()
                optimizer_gnn_cls.step()

                # learn h_s by h_v
                ind_loss, rebuiled_x = model(dataset_tr, criterion, step=3, id=i)
                rebuild_loss = F.mse_loss(rebuiled_x, x)
                env_feature_loss = rebuild_loss + ind_loss * args.idp
                optimizer_ir_learner.zero_grad()
                env_feature_loss.backward()
                optimizer_ir_learner.step()

                #  update partition maximize penalty
                if epoch % args.pud_ro_step == 0:
                    Mean_penalty = model(dataset_tr, criterion, step=4, id=i)
                    Mean_penalty = -Mean_penalty
                    optimizer_env_cls.zero_grad()
                    Mean_penalty.backward()
                    optimizer_env_cls.step()

            accs, test_outs = evaluate_whole_graph(args, model, datasets_tr, datasets_val, datasets_te,
                                                         eval_func)
            logger.add_result(run, accs)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Mean Loss: {Mean:.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)
        print("****************preparing end***************")
    for epoch in range(args.epochs):
        model.train()
        if args.method == 'erm':
            optimizer.zero_grad()
            loss = model(datasets_tr, criterion)
            loss.backward()
            optimizer.step()
        elif args.method == 'iene':
            for i, dataset_tr in enumerate(datasets_tr):
                # minimize dif_cls w
                dif_cls_loss = model(dataset_tr, criterion, step=1, id=i)
                optimizer_cls.zero_grad()
                dif_cls_loss.backward()
                optimizer_cls.step()
                # minimize cls w with dif_cls
                Target, Mean, Var, Penalty = model(dataset_tr, criterion, step=5, id=i)
                cls_loss = Target
                optimizer_gnn_cls.zero_grad()
                cls_loss.backward()
                optimizer_gnn_cls.step()
                # x/a = maximize penalty
                if epoch % args.pud_a_step == 0:
                    model(dataset_tr, criterion, step=6, id=i)

        accs, test_outs = evaluate_whole_graph(args, model, datasets_tr, datasets_val, datasets_te, eval_func)
        logger.add_result(run, accs)

        if epoch % args.display_step == 0:
            if args.method == 'erm':
                print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * accs[0]:.2f}%, '
                  f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)
            elif args.method == 'iene':
                print(f'Epoch: {epoch:02d}, '
                      f'Mean Loss: {Mean:.4f}, '
                      f'Var Loss: {Var:.4f}, '
                      f'env_cls_penalty : {Penalty :.4f}, '
                      f'Train: {100 * accs[0]:.2f}%, '
                      f'Valid: {100 * accs[1]:.2f}%, ')
                test_info = ''
                for test_acc in accs[2:]:
                    test_info += f'Test: {100 * test_acc:.2f}% '
                print(test_info)

    logger.print_statistics(run)

### Save results ###
results = logger.print_statistics()
filename = f'./results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    log = f"{args.method}," + f"{args.gnn},"
    for i in range(results.shape[1]):
        r = results[:, i]
        log += f"{r.mean():.3f} ± {r.std():.3f},"
    write_obj.write(log + f"\n")
