import os
import sys
sys.path.append('..')
from utils.configure_loader import load_configure

# After modularization, select the modules.
# First, considering the accuracy, the loss of accuracy should less than 1%.
# Then, considering the number of kernels.
# estimator_indices = [1, 3, 4, 6, 8, 10, 11, 14, 15, 16]
estimator_indices = [9]
best_epoch = [6]  # simcnn_cifar10

model = 'rescnn'
dataset = 'cifar100'
lr_head = 0.01
lr_modularity = 0.001
alpha = 0.1  # for the weighted sum of loss1 and loss2
batch_size = 64

configs = load_configure(model, dataset)

for i, epoch in enumerate(best_epoch):
    idx = estimator_indices[i]
    configs.set_estimator_idx(idx)
    module_save_dir = f'{configs.module_save_dir}/lr_{lr_head}_{lr_modularity}_alpha_{alpha}'

    cmd = f'cp {module_save_dir}/epoch_{epoch}.pth ' \
          f'{configs.module_save_dir}/estimator_{idx}.pth'
    os.system(cmd)
    print(cmd)