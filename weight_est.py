import numpy as np 
import pickle
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import softmax
from heteroskedastic_cifar import HETEROSKEDASTICCIFAR10, HETEROSKEDASTICCIFAR100
import argparse

parser = argparse.ArgumentParser(description='PyTorch Weight Estimation')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('--exp_str', default='example', help='to indicate which experiment it is')
parser.add_argument('--statspath', default='./log/estimate_cifar10_resnet32_hetero_0.5_0_example/stats0.pkl', help='path to the saved file')
parser.add_argument('--mislabel_type', type=str, default='hetero')
parser.add_argument('--mislabel_ratio', type=float, default=0.5)
parser.add_argument('--imb_type', type=str, default=None)
parser.add_argument('--imb_ratio', type=float, default=0.1)
parser.add_argument('--rand-number', type=int, default=0,
                    help="fix random number for data sampling")

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.statspath, 'rb') as fin:
        data = pickle.load(fin)
    
    if args.dataset == 'cifar10':
        dataset = HETEROSKEDASTICCIFAR10(root='./data', mislabel_type=args.mislabel_type, mislabel_ratio=args.mislabel_ratio, imb_type=args.imb_type, imb_ratio=args.imb_ratio, rand_number=args.rand_number, download=True)
        num_cls = 10
    else:
        dataset = HETEROSKEDASTICCIFAR100(root='./data', mislabel_type=args.mislabel_type, mislabel_ratio=args.mislabel_ratio, rand_number=args.rand_number, download=True)
        num_cls = 100
    targets_np = np.array(dataset.targets)

    _, cnts = np.unique(targets_np, return_counts=True)
    cls_err = 1 - data
    cls_reg = cls_err ** 0.6 / cnts ** 0.4
    cls_reg = cls_reg / np.max(cls_reg)

    np.save('./data/%s_%s_weights.npy'%(args.dataset, args.exp_str), cls_reg)
        
        