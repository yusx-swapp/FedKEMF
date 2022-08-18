import json
import logging
import os
import torch

import datetime

import numpy as np

from lib.fl import local_update, cloud_update, compute_accuracy

from utils.load_neural_networks import init_nets, init_fl
from utils.log_utils import mkdirs
from utils.parameters import get_parameter
from utils.data.prepare_data import partition_data, get_dataloader, sample_dataloader
from utils.save_model import save_checkpoint







if __name__ == '__main__':

    args = get_parameter()


    ########################################create log file###########################################################

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path=args.model+'_'+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+'arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = args.model+'_'+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+'_experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    device = torch.device(args.device)
    logger.info(device)
    ###################################################################################################
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")

    '''
    prepare data
    '''
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)

    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_fl(args.n_parties,args.model, args)
    global_models, global_model_meta_data, global_layer_type = init_nets( 1, args.k_model,args)
    global_model = global_models[0]

    global_para = global_model.module.state_dict()


    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round)+"#" * 100)

        # select clients
        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:int(args.n_parties * args.sample)]


        #local updates:
        _, k_nets, lr = local_update(nets,global_model, selected, args, net_dataidx_map, test_dl_global,logger, lr=args.lr, device=device)


        #cloud updates
        train_dl_global, test_dl_global = sample_dataloader(args.dataset,
                                                          args.datadir,
                                                          args.batch_size,
                                                          32)
        lr_g = cloud_update(k_nets,global_model,train_dl_global,test_dl_global,args.lr_g, args.epochs,device)

        acc_g =compute_accuracy(global_model,test_dl_global,device=device)
        print("communication round %d global model test acc %f" % (round, acc_g))
        logger.info("communication round %d global model test acc %f" % (round, acc_g))






    save_checkpoint({
        'state_dict': global_model.module.state_dict() if isinstance(global_model,
                                                                     torch.nn.DataParallel) else global_model.state_dict(),
        # 'acc': test_acc,

    }, checkpoint_dir=args.ckp_dir)


'''
python knowlege_aggregation.py \
--dataset=cifar100 \
--lr=0.01 \
--batch-size=64 \
--epochs=20 \
--n_parties=10 \
--device='cuda' \
--logdir='./logs_test/'  \
--sample=1 \

--partition=noniid-labeldir \

--beta=0.1 \
--datadir='./data/' \
--noise=0 \
--init_seed=0 \



pip install sklearn
'''

