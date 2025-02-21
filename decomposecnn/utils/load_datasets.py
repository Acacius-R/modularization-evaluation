import copy
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader


def get_dataset_loader(dataset_name):
    if dataset_name == 'cifar10':
        load_dataset = _load_cifar10
    elif dataset_name == 'svhn':
        load_dataset = _load_svhn
    elif dataset_name == 'cifar10_svhn':
        load_dataset = _load_inter_dataset
    elif dataset_name == 'svhn_5':
        load_dataset = _load_svhn_5
    else:
        raise ValueError
    return load_dataset


def _load_cifar10(dataset_dir, is_train, labels=None, batch_size=64, num_workers=0, pin_memory=False,
                  is_random=True, part_train=-1):
    """airplane	 automobile	 bird	 cat	 deer	 dog	 frog	 horse	 ship	 truck"""
    if labels is None:
        labels = list(range(10))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    normalize])
    if is_train:
        if is_random:
            transform = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            normalize])

        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, transform=transform)
        train_targets = np.array(train.targets)
        idx = np.isin(train_targets, labels)
        target_label = train_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        train.targets = trans_label
        train.data = train.data[idx]

        idx = list(range(len(train)))
        np.random.seed(1009)
        np.random.shuffle(idx)
        train_idx = idx[: int(0.8 * len(idx))]
        valid_idx = idx[int(0.8 * len(idx)):]

        if part_train > 0:
            train_idx = train_idx[:part_train]

        train_set = copy.deepcopy(train)
        train_set.targets = [train.targets[idx] for idx in train_idx]
        train_set.data = train.data[train_idx]
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=is_random,
                                  num_workers=num_workers, pin_memory=pin_memory)

        valid_set = copy.deepcopy(train)
        valid_set.targets = [train.targets[idx] for idx in valid_idx]
        valid_set.data = train.data[valid_idx]
        valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, valid_loader

    else:
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
                                            transform=transform)

        test_targets = np.array(test.targets)
        idx = np.isin(test_targets, labels)
        target_label = test_targets[idx].tolist()
        trans_label = [labels.index(i) for i in target_label]
        test.targets = trans_label
        test.data = test.data[idx]

        test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        return test_loader