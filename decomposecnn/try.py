import argparse
from ast import mod
import copy
import torch
from tqdm import tqdm
from utils.checker import check_dir
from utils.load_datasets import get_dataset_loader
from utils.model_loader import load_model
from concern_identification import CI
from concern_modularization import back_track
from utils.tools import extract_weights
from tqdm import tqdm
model_name ='resnet20'
dataset_name = 'cifar10'
model = load_model(model_name,10)
model.load_state_dict(torch.load(f'./models/{model_name}_{dataset_name}.pth'))
load_dataset = get_dataset_loader(dataset_name)
dataset_dir = '/home/rq/modularization/decomposecnn/datasets'
train_loader, val_loader = load_dataset(dataset_dir, is_train=True,
                                            batch_size=1, num_workers=1, pin_memory=True)

for batch_inputs, batch_labels in train_loader:
    outputs = model(batch_inputs)
    print(outputs)
    break