import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, SVHN,CIFAR100


def load_cifar10(dataset_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR10(root=dataset_dir, train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def load_cifar100_single_target_class(dataset_dir, batch_size, num_workers, target_class):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    
    target_class = target_class[0]
    # Load train dataset
    train_dataset = CIFAR100(root=dataset_dir, train=True, transform=transform, download=True)
    train_targets = np.array(train_dataset.targets)
    
    # Select target class samples
    target_idx = np.where(train_targets == target_class)[0]
    target_count = len(target_idx)
    
    # Select non-target samples (balanced count)
    non_target_idx = np.where(train_targets != target_class)[0]
    sampled_non_target_idx = np.random.choice(non_target_idx, target_count, replace=False)
    
    # Combine indices and shuffle
    selected_idx = np.concatenate([target_idx, sampled_non_target_idx])
    np.random.shuffle(selected_idx)
    
    # Update dataset
    train_dataset.data = train_dataset.data[selected_idx]
    train_dataset.targets = [1 if train_targets[i] == target_class else 0 for i in selected_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    
    # Load test dataset
    test_dataset = CIFAR100(root=dataset_dir, train=False, transform=transforms.Compose([
        transforms.ToTensor(), normalize]), download=True)
    test_targets = np.array(test_dataset.targets)
    
    # Select target class samples
    target_idx = np.where(test_targets == target_class)[0]
    target_count = len(target_idx)
    
    # Select non-target samples (balanced count)
    non_target_idx = np.where(test_targets != target_class)[0]
    sampled_non_target_idx = np.random.choice(non_target_idx, target_count, replace=False)
    
    # Combine indices and shuffle
    selected_idx = np.concatenate([target_idx, sampled_non_target_idx])
    np.random.shuffle(selected_idx)
    
    # Update dataset
    test_dataset.data = test_dataset.data[selected_idx]
    test_dataset.targets = [1 if test_targets[i] == target_class else 0 for i in selected_idx]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader

def load_svhn_single_target_class(dataset_dir, batch_size, num_workers, target_class):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    
    target_class = target_class[0]
    # Load train dataset
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform, download=True)

    
    train_targets = np.array(train_dataset.labels)
    
    # Select target class samples
    target_idx = np.where(train_targets == target_class)[0]
    target_count = len(target_idx)
    
    # Select non-target samples (balanced count)
    non_target_idx = np.where(train_targets != target_class)[0]
    sampled_non_target_idx = np.random.choice(non_target_idx, target_count, replace=False)
    
    # Combine indices and shuffle
    selected_idx = np.concatenate([target_idx, sampled_non_target_idx])
    np.random.shuffle(selected_idx)
    
    # Update dataset
    train_dataset.data = train_dataset.data[selected_idx]
    train_dataset.labels = [1 if train_targets[i] == target_class else 0 for i in selected_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    
    # Load test dataset
    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)
    test_targets = np.array(test_dataset.labels)
    
    # Select target class samples
    target_idx = np.where(test_targets == target_class)[0]
    target_count = len(target_idx)
    
    # Select non-target samples (balanced count)
    non_target_idx = np.where(test_targets != target_class)[0]
    sampled_non_target_idx = np.random.choice(non_target_idx, target_count, replace=False)
    
    # Combine indices and shuffle
    selected_idx = np.concatenate([target_idx, sampled_non_target_idx])
    np.random.shuffle(selected_idx)
    
    # Update dataset
    test_dataset.data = test_dataset.data[selected_idx]
    test_dataset.labels = [1 if test_targets[i] == target_class else 0 for i in selected_idx]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader
def load_cifar10_single_target_class(dataset_dir, batch_size, num_workers, target_class):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    
    target_class = target_class[0]
    # Load train dataset
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
    train_targets = np.array(train_dataset.targets)
    
    # Select target class samples
    target_idx = np.where(train_targets == target_class)[0]
    target_count = len(target_idx)
    
    # Select non-target samples (balanced count)
    non_target_idx = np.where(train_targets != target_class)[0]
    sampled_non_target_idx = np.random.choice(non_target_idx, target_count, replace=False)
    
    # Combine indices and shuffle
    selected_idx = np.concatenate([target_idx, sampled_non_target_idx])
    np.random.shuffle(selected_idx)
    
    # Update dataset
    train_dataset.data = train_dataset.data[selected_idx]
    train_dataset.targets = [1 if train_targets[i] == target_class else 0 for i in selected_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    
    # Load test dataset
    test_dataset = CIFAR10(root=dataset_dir, train=False, transform=transforms.Compose([
        transforms.ToTensor(), normalize]), download=True)
    test_targets = np.array(test_dataset.targets)
    
    # Select target class samples
    target_idx = np.where(test_targets == target_class)[0]
    target_count = len(target_idx)
    
    # Select non-target samples (balanced count)
    non_target_idx = np.where(test_targets != target_class)[0]
    sampled_non_target_idx = np.random.choice(non_target_idx, target_count, replace=False)
    
    # Combine indices and shuffle
    selected_idx = np.concatenate([target_idx, sampled_non_target_idx])
    np.random.shuffle(selected_idx)
    
    # Update dataset
    test_dataset.data = test_dataset.data[selected_idx]
    test_dataset.targets = [1 if test_targets[i] == target_class else 0 for i in selected_idx]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader

def load_cifar10_target_class(dataset_dir, batch_size, num_workers, target_classes):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform)
    train_targets = np.array(train_dataset.targets)
    idx = np.isin(train_targets, target_classes)
    target_label = train_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    train_dataset.targets = trans_label
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR10(root=dataset_dir, train=False,
                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_targets = np.array(test_dataset.targets)
    idx = np.isin(test_targets, target_classes)
    target_label = test_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    test_dataset.targets = trans_label
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_svhn(dataset_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_svhn_target_class(dataset_dir, batch_size, num_workers, target_classes):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform)
    train_labels = train_dataset.labels
    idx = np.isin(train_labels, target_classes)
    target_labels = train_labels[idx].tolist()
    trans_labels = np.array([target_classes.index(i) for i in target_labels])
    train_dataset.labels = trans_labels
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_labels = test_dataset.labels
    idx = np.isin(test_labels, target_classes)
    target_labels = test_labels[idx].tolist()
    trans_labels = np.array([target_classes.index(i) for i in target_labels])
    test_dataset.labels = trans_labels
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def load_cifar100(dataset_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR100(root=dataset_dir, train=True, transform=transform, download=True)
    train_targets = np.array(train_dataset.targets)
    target_classes = list(range(10))  # 仅加载前十种类别
    idx = np.isin(train_targets, target_classes)
    target_label = train_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    train_dataset.targets = trans_label
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR100(root=dataset_dir, train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_targets = np.array(test_dataset.targets)
    idx = np.isin(test_targets, target_classes)
    target_label = test_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    test_dataset.targets = trans_label
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_cifar100_target_class(dataset_dir, batch_size, num_workers, target_classes):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR100(root=dataset_dir, train=True, transform=transform)
    train_targets = np.array(train_dataset.targets)
    idx = np.isin(train_targets, target_classes)
    target_label = train_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    train_dataset.targets = trans_label
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR100(root=dataset_dir, train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_targets = np.array(test_dataset.targets)
    idx = np.isin(test_targets, target_classes)
    target_label = test_targets[idx].tolist()
    trans_label = [target_classes.index(i) for i in target_label]
    test_dataset.targets = trans_label
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
if __name__ == '__main__':
    for tc in range(10):
        dataset = load_svhn_target_class(dataset_dir='../data/dataset/svhn', batch_size=128,
                                         num_workers=0, target_classes=[tc])
        print(f'tc_{tc} = {len(dataset[0])}')
