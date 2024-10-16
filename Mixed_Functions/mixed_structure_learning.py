import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from Mixed_Functions.dataset import load_k_fold
from Backbone_Models.global_GCN import GCN
from Backbone_Models.global_GAT import GAT
from Backbone_Models.global_GIN import GIN
from Backbone_Models.global_GraphSNN import GraphSNN
from Backbone_Models.global_I2GNN import I2GNN
from Backbone_Models.Hierarchical_SAGPool import SAGPool
from Backbone_Models.Hierarchical_ASAP import ASAP
from Backbone_Models.Hierarchical_DMoN import DMoN
from Backbone_Models.Hierarchical_k_MISPool import k_MISPool


def get_backbone_model(backbone, args):
    if backbone == 'GCN':
        model = GCN(args)
    elif backbone == 'GAT':
        model = GAT(args)
    elif backbone == 'GIN':
        model = GIN(args)
    elif backbone == 'GraphSNN':
        model = GraphSNN(args)
    elif backbone == 'I2GNN':
        model = I2GNN(args)
    elif backbone == 'SAGPool':
        model = SAGPool(args)
    elif backbone == 'ASAP':
        model = ASAP(args)
    elif backbone == 'DMoN':
        model = DMoN(args)
    elif backbone == 'k_MISPool':
        model = k_MISPool(args)

    return model


def mix_operation(data, substructure_list, mix_probab_learnable, mix_training=True):
    if mix_training == True:
        mix_probab_softmax = F.softmax(mix_probab_learnable[0], dim=-1)[0]
    else:
        mix_probab_softmax = F.softmax(mix_probab_learnable[0], dim=-1)[0].detach().cpu()

    for substructure_method, mix_pro in zip(substructure_list, mix_probab_softmax):
        substructure_feature = data[f'{substructure_method}_feature'] * mix_pro
        data.x = torch.cat((data.x, substructure_feature), dim=1)

    return data


def eval_acc(eval_data, model, substructure_list, mix_probab_learnable, args):
    acc = 0
    with torch.no_grad():
        for data in eval_data:
            model.eval()
            data = data.to(args.device)

            data_dc = copy.deepcopy(data)
            if args.use_substructure_feature == True:
                data_dc = mix_operation(data_dc, substructure_list, mix_probab_learnable, mix_training=model.training)

            probas_pred, ground_truth = model(data_dc), data_dc.y.view(-1)
            probas_pred = probas_pred if args.backbone not in ['DMoN'] else probas_pred[0]

            acc += probas_pred.max(1)[1].eq(ground_truth.view(-1)).sum().item()
        acc /= len(eval_data.dataset)

    return acc



def Mixed_Structure_Learning(graph_data, substructure_list, args):

    valid_accs = []
    test_accs = []

    k_folds_data = load_k_fold(graph_data, args.folds, args.batch_size, args.backbone)
    argmax_list = []

    if args.use_substructure_feature == True:
        best_mix_probab_list = []
        print(32 * "=")
        print("Start Mixed Structure Learning")

    for fold, fold_data in enumerate(k_folds_data):

        model = get_backbone_model(args.backbone, args).to(args.device)
        if args.use_substructure_feature == True:
            mix_probab_learnable = [Variable(1e-3 * torch.randn(1, len(substructure_list)).to(args.device), requires_grad=True)]
            optimizer = optim.Adam([{'params': model.parameters()},
                                    {'params': mix_probab_learnable}],
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
            best_mix_probab = None
        else:
            mix_probab_learnable = None
            optimizer = optim.Adam([{'params': model.parameters()}],
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
                                                         eta_min=args.learning_rate/10.0)
        loss_fn = F.nll_loss

        print('###fold {}, train/val/test:{},{},{}'.format(fold+1, len(fold_data[-3].dataset),len(fold_data[-2].dataset),len(fold_data[-1].dataset)))
        max_acc = 0
        max_index = 0


        for epoch in range(1, args.epochs + 1):
            for i, data in enumerate(fold_data[-3]): # train dataloader
                model.train()

                data = data.to(args.device)

                data_dc = copy.deepcopy(data)
                if args.use_substructure_feature == True:
                    data_dc = mix_operation(data_dc, substructure_list, mix_probab_learnable, mix_training=model.training)

                probas_pred, ground_truth = model(data_dc), data_dc.y.view(-1)
                if args.backbone not in ['DMoN']:
                    loss = loss_fn(probas_pred, ground_truth)
                else:
                    loss = loss_fn(probas_pred[0], ground_truth)
                    for aux_loss in probas_pred[1]:
                        loss = loss + aux_loss

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()


            scheduler.step()

            valid_data = fold_data[-2] # val dataloader
            valid_acc = eval_acc(valid_data, model, substructure_list, mix_probab_learnable, args)
            valid_accs.append(valid_acc)

            test_data = fold_data[-1] # test dataloader
            test_acc = eval_acc(test_data, model, substructure_list, mix_probab_learnable, args)
            test_accs.append(test_acc)

            if valid_acc >= max_acc:
                max_acc = valid_acc
                max_index = epoch-1
                if args.use_substructure_feature == True:
                    m_pro = mix_probab_learnable[0][0].detach().cpu().tolist()
                    m_pro_s = F.softmax(mix_probab_learnable[0], dim=-1)[0].detach().cpu().tolist()
                    best_mix_probab = (m_pro, m_pro_s)

            if epoch % 10 == 0:
                print('fold:{}, epoch:{}, valid_acc:{:.4f}, test_acc:{:.4f}'.format(fold+1,epoch,valid_acc,test_acc))

        argmax_list.append(max_index)
        if args.use_substructure_feature == True:
            best_mix_probab_list.append(best_mix_probab)

    if args.use_substructure_feature == True:
        print(32 * "=")
        print("End Mixed Structure Learning")

    valid_accs = torch.tensor(valid_accs).view(args.folds, args.epochs)
    test_accs = torch.tensor(test_accs).view(args.folds, args.epochs)

    # max_valid_acc
    valid_accs_argmax = valid_accs[torch.arange(args.folds, dtype=torch.long), argmax_list] * 100
    valid_acc_mean = round(valid_accs_argmax.mean().item(), 2)
    test_accs_argmax = test_accs[torch.arange(args.folds, dtype=torch.long), argmax_list] * 100
    test_acc_mean = round(test_accs_argmax.mean().item(), 2)
    test_acc_std = round(test_accs_argmax.std().item(), 2)
    # print('test_accs:', test_accs_argmax)

    return valid_acc_mean, test_acc_mean, test_acc_std
