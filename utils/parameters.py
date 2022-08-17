import argparse

def get_parameter():
    parser = argparse.ArgumentParser(description='gnnrl search script')


    # parser.add_argument('--data_root', default='data', type=str, help='dataset path')

    parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
    parser.add_argument('--output', default='./logs', type=str, help='')
    # parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    # parser.add_argument('--n_gpu', default=4, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')

    parser.add_argument('--log_dir', default='./logs', type=str, help='log dir')
    parser.add_argument('--comm_round', type=int, default=500, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    # parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')

    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    # parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    # parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet32', help='neural network used in training')
    parser.add_argument('--k_model', type=str, default='resnet20', help='neural network used in training')

    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    parser.add_argument('--logdir', type=str, required=True, default="./logs/", help='Log directory path')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr_g', type=float, default=0.01, help='learning rate (default: 0.01)')

    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--ckp_dir', type=str, default='ckp_dir/',  help='number of workers in a distributed cluster')

    return parser.parse_args()
