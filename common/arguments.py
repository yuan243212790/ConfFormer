import argparse
import os
import math
import time
import torch
import socket
import sys
import logging

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='mixste_confformer', type=str)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument('--d_hid', default=1024, type=int)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--pad', type=int, default=121)
    parser.add_argument('--token_num', default=81, type=int)
    parser.add_argument('--layer_index', default=3, type=int)
    parser.add_argument('--dataset', type=str, default="h36m")
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb_with_confidence', type=str)
    parser.add_argument('--res_w_3dhp', type=int, default=2048)
    parser.add_argument('--res_h_3dhp', type=int, default=2048)
    parser.add_argument('--data_augmentation', type=int, default=1)
    parser.add_argument('--reverse_augmentation', type=str2bool, default=False)
    parser.add_argument('--test_augmentation', type=str2bool, default=True)
    parser.add_argument('--fast_test', action='store_true',
                        help='Use sparse 3DPW evaluation during training for speed. Not official per-frame evaluation.')
    parser.add_argument('--final_full_test', type=str2bool, default=True,
                        help='After training, run one full test with eval_stride=1 and test augmentation enabled.')
    parser.add_argument('--eval_stride', type=int, default=1,
                        help='3DPW test center-frame stride. 1 is per-frame official-style evaluation.')
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--actions', default='*', type=str)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('--stride', default=243, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nepoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--lr_decay_large', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=200)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('-lrd', '--lr_decay', default=0.99, type=float)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--previous_dir', type=str, default='')
    parser.add_argument('--n_joints', type=int, default=17)
    parser.add_argument('--out_joints', type=int, default=17)
    parser.add_argument('--out_all', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--previous_best', type=float, default= math.inf)
    parser.add_argument('--previous_name', type=str, default='')

    args = parser.parse_args()
    args.root_path = os.path.join(args.root_path, '')
    args.dataset = args.dataset.lower()

    if args.test:
        args.train = 0

    args.pad = (args.frames-1) // 2
    if args.fast_test:
        args.eval_stride = args.frames
        args.test_augmentation = False

    args.root_joint = 0
    if args.dataset == 'h36m':
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'
    elif args.dataset == 'humaneva15':
        args.subjects_train = 'Train/S1,Train/S2,Train/S3'
        args.subjects_test = 'Validate/S1,Validate/S2,Validate/S3'
        args.n_joints = 15
        args.out_joints = 15
    elif args.dataset == '3dhp':
        args.subjects_train = 'S1,S2,S3,S4,S5,S6,S7,S8'
        args.subjects_test = 'TS1,TS2,TS3,TS4,TS5,TS6'
        args.n_joints = 17
        args.out_joints = 17
        args.root_joint = 14
    elif args.dataset in ['3dpw', 'pw3d']:
        args.subjects_train = 'train,validation'
        args.subjects_test = 'test'
        args.n_joints = 17
        args.out_joints = 17
        args.root_joint = 0

    if args.train:
        logtime = time.strftime('%m%d_%H%M_%S_')
        args.checkpoint = 'checkpoint/' + logtime + '%d'%(args.frames)
        os.makedirs(args.checkpoint, exist_ok=True)

        args_write = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))

        file_name = os.path.join(args.checkpoint, 'configs.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args_write.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)

    return args
