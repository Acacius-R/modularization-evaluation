import os

# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.

# train CNN models
# model = 'resnet18'
# dataset = 'svhn'
# target_classes = [0,1,2,3,4,5,6,7,8,9]

# for i in range(10):
#     cmd = f'python modularizer.py ' \
#           f'--model {model} --dataset {dataset} --target_classes {i}  >> {model}_{dataset}_kernel_log.txt'
#     print(cmd)
#     os.system(cmd)

models = ['resnet18','vgg16']
datasets = ['cifar10','svhn','cifar100']
for model in models:
    for dataset in datasets:
            if model == 'vgg16' and dataset =='svhn':
                  continue
            cmd = f'python modularizer.py ' \
          f'--model {model} --dataset {dataset} --target_class 0'
            print(cmd)
            os.system(cmd)