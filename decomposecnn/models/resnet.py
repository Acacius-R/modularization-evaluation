import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
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
        self.layer_info = []
        self.conv1 =self._add_layer(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),'conv')
        self.bn1 = self._add_layer(nn.BatchNorm2d(planes),'bn')
        self._add_layer(nn.ReLU(),'relu')
        self.conv2 = self._add_layer(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),'conv')
        self.bn2 = self._add_layer(nn.BatchNorm2d(planes),'bn')


        self.shortcut = nn.Sequential()
        self._add_layer(nn.Identity(),'shortcut')
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x: 
                    F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                self._add_layer(self.shortcut, 'pad')
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    self._add_layer(nn.Conv2d(in_planes, self.expansion*planes, 
                                    kernel_size=1, stride=stride, bias=False), 'conv'),
                    self._add_layer(nn.BatchNorm2d(self.expansion*planes), 'bn')
                )
        self._add_layer(nn.Identity(),'add')
        self._add_layer(nn.ReLU(),'relu')
    def _add_layer(self, layer, layer_type):
        """添加层并记录元数据"""
        self.layer_info.append({
            'type': layer_type,
            'object': layer,
            'params': {
                'weight': getattr(layer, 'weight', None),
                'bias': getattr(layer, 'bias', None)
            }
        })
        return layer
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.layer_info = []
        self.conv1 = self._add_layer(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),'conv')
        self.bn1 = self._add_layer(nn.BatchNorm2d(16),'bn')
        self._add_layer(nn.ReLU(),'relu')
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.avgpool = self._add_layer(nn.AdaptiveAvgPool2d(1), 'pool')
        self.flatten = self._add_layer(nn.Flatten(), 'flatten')
        self.linear = self._add_layer(nn.Linear(64, num_classes), 'dense')

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            blk = block(self.in_planes, planes, stride)
            self.layer_info.extend(blk.layer_info)
            layers.append(blk)
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def _add_layer(self, layer, layer_type):
        """添加层并记录元数据"""
        self.layer_info.append({
            'type': layer_type,
            'object': layer,
            'params': {
                'weight': getattr(layer, 'weight', None),
                'bias': getattr(layer, 'bias', None)
            }
        })
        return layer
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
    


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])