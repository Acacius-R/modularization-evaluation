import argparse
import copy
import random
from models.vgg_masked import cifar10_vgg16_bn as mt_vgg16_model
from modules_arch.vgg_module_v2 import cifar10_vgg16_bn as vgg16_module
from models.vgg import cifar10_vgg16_bn as st_vgg16_model

from models.resnet_masked import ResNet18 as mt_ResNet18_model
from modules_arch.resnet_module_v2 import ResNet18 as ResNet18_module
from models.resnet import ResNet18 as st_ResNet18_model

from models_cnnsplitter.simcnn_masked import SimCNN as mt_simcnn_model
from modules_arch.simcnn_module import SimCNN as simcnn_module
from models_cnnsplitter.simcnn import SimCNN as st_simcnn_model

from models_cnnsplitter.rescnn_masked import ResCNN as mt_rescnn_model
from modules_arch.rescnn_module import ResCNN as rescnn_module
from models_cnnsplitter.rescnn import ResCNN as st_rescnn_model

import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from configs import Configs
from dataset_loader import load_cifar10, load_cifar10_target_class, load_svhn, load_svhn_target_class,load_cifar100_target_class,load_cifar100,\
                            load_cifar10_single_target_class,load_svhn_single_target_class,load_cifar100_single_target_class
import thop
import csv
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['vgg16', 'resnet18', 'simcnn', 'rescnn'], required=True)
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'svhn','cifar100'], required=True)
    args = parser.parse_args()
    return args

def test(model, test_loader):
    epoch_acc = []
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_inputs)
            pred = torch.argmax(outputs, dim=1)
            correct = (pred == batch_labels).squeeze()
            for i in range(len(batch_labels)):
                label = batch_labels[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            acc = torch.sum(pred == batch_labels)
            epoch_acc.append(torch.div(acc, batch_labels.shape[0]))
    print(f"\nTest Accuracy: {sum(epoch_acc) / len(epoch_acc) * 100:.2f}%")
    for i in range(10):
        if class_total[i] > 0:
            print(f"Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Accuracy of class {i}: N/A (no samples)")



if __name__ == '__main__':
    args = get_args()
    DEVICE = torch.device('cuda')
    model_name = args.model
    dataset_name = args.dataset
    batch_size=128
    configs = Configs()
    save_dir = f'{configs.data_dir}/{model_name}_{dataset_name}'
    st_model_save_path = f'{save_dir}/standard_model_lr_0.05_bz_128.pth'
    
    if dataset_name == 'cifar10':
        train_loader, test_loader = load_cifar10(configs.dataset_dir, batch_size=batch_size, num_workers=2)
    elif dataset_name == 'svhn':
        train_loader, test_loader = load_svhn(f'{configs.dataset_dir}/svhn', batch_size=batch_size, num_workers=2)
    elif dataset_name == 'cifar100':
        train_loader, test_loader = load_cifar100(configs.dataset_dir, batch_size=batch_size, num_workers=2)
    else:
        raise ValueError
    
    if model_name == 'vgg16':
        st_model = st_vgg16_model(pretrained=False).to(DEVICE)
    elif model_name == 'resnet18':
        st_model = st_ResNet18_model().to(DEVICE)
    elif model_name == 'simcnn':
        st_model = st_simcnn_model().to(DEVICE)
    elif model_name == 'rescnn':
        st_model = st_rescnn_model().to(DEVICE)
    else:
        raise ValueError
    st_model.load_state_dict(torch.load(st_model_save_path, map_location=DEVICE))
    test(st_model, test_loader)
