import os
import numpy as np
import torch
import random
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from Mixed_Functions.dataset import load_data, generate_GraphSNN_adj, generate_I2GNN_data
from Mixed_Functions.mixed_structure_learning import Mixed_Structure_Learning


graph_classification_dataset=['DD','MUTAG','PROTEINS','NCI1','NCI109','IMDB-BINARY','REDDIT-BINARY', 'BZR', 'COX2', 'IMDB-MULTI', 'COLORS-3', 'COLLAB', 'REDDIT-MULTI-5K', 'KKI', 'OHSU', 'PROTEINS_full', 'Peking_1', 'github_stargazers']

def arg_parse():
    str2bool = lambda x: x.lower() == "true"

    parser = argparse.ArgumentParser("MixSL.")
    parser.add_argument('--data_name', type=str, default='MUTAG', help='location of the data corpus')
    parser.add_argument('--backbone', type=str, default='GIN', help='the choice of backbone model for validing substructure feature.',
                        choices=['GCN', 'GAT', 'GIN', 'GraphSNN', 'I2GNN', 'SAGPool', 'ASAP', 'DMoN', 'k_MISPool'])
    parser.add_argument('--use_substructure_feature', type=str2bool, default=True,
                        help='whether to use substructure features. '
                             'If True, the backbone will use substructure features. '
                             'If False, the backbone will not use substructure features.',
                        choices=[True, False])
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='default hidden_dim for gnn model')
    parser.add_argument('--dropout', type=float, default=0.0, help='default dropout for gnn model')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--folds', type=int, default=5, help='cross validation')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = arg_parse()
set_seed(args.seed)

device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
args.device = device

args.graph_classification_dataset = graph_classification_dataset
if args.data_name in args.graph_classification_dataset:
    print(35 * "=" + " starting " + 35 * "=")
    print("choose backbone:", args.backbone)

    graph_data, num_classes, num_nodes = load_data(args.data_name)
    args.num_classes = num_classes

    if args.backbone == 'GraphSNN':
        graph_data, N_nodes_max = generate_GraphSNN_adj(graph_data)
        args.N_nodes_max = N_nodes_max
    elif args.backbone == 'I2GNN':
        graph_data = generate_I2GNN_data(graph_data)

    
    if args.use_substructure_feature == True:
        x_dim = graph_data[0].x.shape[1]
        substructure_list = ["RW", "evc", "clu", "cliq"]
        substructure_dims = [graph_data[0][f'{substructure_method}_feature'].shape[1] for substructure_method in substructure_list]
        num_features = x_dim + sum(substructure_dims)

        args.num_features = num_features
        print('substructure_list:', substructure_list)
    else:
        x_dim = graph_data[0].x.shape[1]
        num_features = x_dim

        args.num_features = num_features
        substructure_list = None
        print('not use substructure features')


    ### repeat cross validation X times
    test_repeat = 5
    all_Test_acc = []
    for i in range(test_repeat):
        valid_acc, test_acc, test_acc_std = Mixed_Structure_Learning(graph_data, substructure_list, args=args)
        print("{}-th run in all {} runs || Test_acc:{}±{}".format(i + 1, test_repeat, test_acc, test_acc_std))
        all_Test_acc.append(test_acc)
    all_Test_acc = np.array(all_Test_acc)
    print('final results:{:.2f}±{:.2f}'.format(all_Test_acc.mean(), all_Test_acc.std()))
    print(35 * "=" + " ending " + 35 * "=")
