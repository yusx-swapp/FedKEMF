import copy

import torch
from sklearn.metrics import confusion_matrix
from torch import optim, nn
import numpy as np

from lib.distillate import Distiller
from lib.ensemble import AvgEnsemble
from utils.accuracy import compute_acc
from utils.data.prepare_data import get_dataloader
from utils.loss import LossCalculator
import torch.nn as nn

#TODO learning rate decay
def local_update(nets:dict, g_k:nn.Module, selected, args, net_dataidx_map, test_dl_global,logger,lr = 0.01, device="cpu"):
    avg_acc = 0.0
    avg_kacc = 0.0
    k_nets = []
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        n_epoch = args.epochs



        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        kd_agent = Distiller(lr=lr,epochs=n_epoch,device=device)

        nets_cohort = []
        optimizers = []

        nets_cohort.append(net)
        net_optimizer = optim.SGD(net.parameters(), lr)
        optimizers.append(net_optimizer)

        l_g_k = copy.deepcopy(g_k)
        k_nets.append(l_g_k)
        nets_cohort.append(l_g_k)
        gk_optimizer = optim.SGD(l_g_k.parameters(), lr)
        optimizers.append(gk_optimizer)



        #### if use the local test_dl (also need modify get_dataloader function):
        '''
        nets_cohort, lr_ = kd_agent.mutual_kd(nets_cohort,train_dl_local, test_dl_local,optimizers,s_save_path=None)

        acc =compute_accuracy(nets_cohort[0],test_dl_local,device=device)
        print("net %d final test acc %f" % (net_id, acc))
        logger.info("net %d final test acc %f" % (net_id, acc))
        acc_k = compute_accuracy(nets_cohort[1],test_dl_local,device=device)
        print("net %d knowledge network test acc %f" % (net_id, acc_k))
        logger.info("net %d knowledge network test acc %f" % (net_id, acc_k))
        '''


        #### if use the global test_dl:

        nets_cohort, lr_ = kd_agent.mutual_kd(nets_cohort, train_dl_local, test_dl_global, optimizers, s_save_path=None)

        acc = compute_accuracy(nets_cohort[0], test_dl_global, device=device)
        print("net %d final test acc %f" % (net_id, acc))
        logger.info("net %d final test acc %f" % (net_id, acc))
        acc_k = compute_accuracy(nets_cohort[1], test_dl_global, device=device)
        print("net %d knowledge network test acc %f" % (net_id, acc_k))
        logger.info("net %d knowledge network test acc %f" % (net_id, acc_k))


        avg_acc += acc
        avg_kacc += acc_k

    avg_acc /= len(selected)
    avg_kacc /= len(selected)
    logger.info("avg test acc after local update %f" % avg_acc)
    logger.info("avg knowledge test acc after local update %f" % avg_kacc)

    nets_list = list(nets.values())
    return nets_list,k_nets,lr_

    # raise NotImplementedError

def cloud_update(l_ks,g_k,train_loader,test_loader,lr, n_epoch,device):
    ensemble = AvgEnsemble(l_ks)
    # teh_optimizer = optim.SGD(ensemble.parameters(), lr)
    stu_optimizer = optim.SGD(g_k.parameters(), lr)
    kd_agent = Distiller(lr=lr,epochs=n_epoch,device=device)
    _, _,_, lr = kd_agent.pure_kd(ensemble,g_k,None,stu_optimizer,train_loader,test_loader,device='cuda')
    return lr
def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix

    return correct / float(total)

