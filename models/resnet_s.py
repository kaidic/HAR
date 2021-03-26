'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

class BatchNorm2dMul(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2dMul, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.affine = affine

    def forward(self, x):
        bn_out = self.bn(x)
        if self.affine:
            out = self.gamma[None, :, None, None] * bn_out + self.beta[None, :, None, None]
        return out, bn_out

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2dMul(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2dMul(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        bn_outputs = []
        out, bn_out = self.bn1(self.conv1(x))
        bn_outputs.append(bn_out)
        out = F.relu(out)
        out, bn_out = self.bn2(self.conv2(out))
        bn_outputs.append(bn_out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out, bn_outputs

class LayerBlock(nn.Module):
    
    def __init__(self, block, in_planes, planes, num_blocks, stride):
        super(LayerBlock, self).__init__()
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        bn_outputs = []
        for layer in self.layers:
            x, bn_output = layer(x)
            bn_outputs.extend(bn_output)
        return x, bn_outputs


class ResNet_s(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2dMul(16)
        self.layer1 = LayerBlock(block, 16, 16, num_blocks[0], stride=1)
        self.layer2 = LayerBlock(block, 16, 32, num_blocks[1], stride=2)
        self.layer3 = LayerBlock(block, 32, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        all_bn_outputs = []
        out, bn_out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out, bn_outputs = self.layer1(out)
        all_bn_outputs.extend(bn_outputs)
        out, bn_outputs = self.layer2(out)
        all_bn_outputs.extend(bn_outputs)
        out, bn_outputs = self.layer3(out)
        all_bn_outputs.extend(bn_outputs)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, all_bn_outputs


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()