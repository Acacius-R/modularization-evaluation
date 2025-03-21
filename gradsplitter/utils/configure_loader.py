import sys
sys.path.append('..')


def load_configure(model_name, dataset_name):
    model_dataset_name = f'{model_name}_{dataset_name}'
    if model_dataset_name == 'resnet18_cifar10':
        from configures.resnet18_cifar10 import Configures
    elif model_dataset_name == 'vgg16_cifar10':
        from configures.vgg16_cifar10 import Configures
    elif model_dataset_name == 'rescnn_cifar10':
        from configures.rescnn_cifar10 import Configures
    elif model_dataset_name == 'rescnn_svhn':
        from configures.rescnn_svhn import Configures
    elif model_dataset_name == 'simcnn_cifar10':
        from configures.simcnn_cifar10 import Configures
    elif model_dataset_name == 'simcnn_svhn':
        from configures.simcnn_svhn import Configures
    elif model_dataset_name == 'vgg16_svhn':
        from configures.vgg16_svhn import Configures
    elif model_dataset_name == 'vgg16_cifar100':
        from configures.vgg16_cifar100 import Configures
    elif model_dataset_name == 'rescnn_cifar100':
        from configures.rescnn_cifar100 import Configures
    elif model_dataset_name == 'simcnn_cifar100':
        from configures.simcnn_cifar100 import Configures
    else:
        raise ValueError()
    configs = Configures()
    return configs