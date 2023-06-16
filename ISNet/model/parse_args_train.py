from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='ISNET_Shape_Matters')
    # parameter for ISNet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False')

    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name:  IRSTD_1K, NUAA-SIRST')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_30_20')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=480,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--weightDecay', default=1e-4,
                        type=float, help='weight decay')

    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.04, metavar='LR',
                        help='learning rate (default: 0.1)')
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')

    args = parser.parse_args()
    # make dir for save result
    args.save_dir = make_dir(args.deep_supervision, args.dataset)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args
