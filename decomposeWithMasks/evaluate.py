import copy
import torch
import argparse
from mask import MaskModel
from torch.optim import SGD
import torch.nn as nn
from torchvision import datasets, transforms
from utils import create_labelwise_dataloaders
from utils import load_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import check_dir
from utils import load_modules
import csv
import thop
import numpy as np
def get_model_weights(model):
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)

def count_nonzero_weights(module):
    nonzero_count = 0
    for layer in list(module.children())[:-1]:  
        for param in layer.parameters():
            nonzero_count += np.count_nonzero(param.data.cpu().numpy())
    return nonzero_count

def generate_random_modules(modules, orig_model):
    """
    生成随机模块。
    
    参数:
        modules (list): 模块列表。
        orig_model (nn.Module): 原始模型。
    
    返回:
        list: 随机模块列表。
    """
    random_modules = []
    orig_weights = get_model_weights(orig_model)
    for module in modules:
        random_module = copy.deepcopy(module)
        nonzero_count = count_nonzero_weights(module)
        indices = np.arange(len(orig_weights))

        selected_indices = np.random.choice(indices, size=nonzero_count, replace=False)
        selected_weights = np.zeros_like(orig_weights)
        selected_weights[selected_indices] = orig_weights[selected_indices]
        start_idx = 0
        for layer in list(random_module.children())[:-1]: 
            for param in layer.parameters():
                end_idx = start_idx + param.data.numel()
                param.data = torch.tensor(selected_weights[start_idx:end_idx].reshape(param.data.shape)).to(param.data.device)
                start_idx = end_idx
        random_modules.append(random_module)
    return random_modules

def jaccard_similarity(list1, list2):
    intersection = len(np.intersect1d(list1, list2))
    union = len(np.union1d(list1, list2))
    
    return float(intersection) / union
def calculate_jaccard_similarity(modules):
    num_modules = len(modules)
    weights_to_compare = [get_model_weights(module) for module in modules]
    
    result = []
    for i in range(num_modules):
        tmp = []
        for j in range(num_modules):
            similarity = jaccard_similarity(weights_to_compare[i], weights_to_compare[j])
            tmp.append(similarity)
        result.append(tmp)

    with open(f'{module_dir}/analaysis/similarity.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(result)
    
    return result

def calculate_flops(test_dataset,modules,model):
    inputs, labels = next(iter(test_dataset))
    inputs = inputs
    model_flops, model_params = thop.profile(model, inputs=(inputs,))
    # print(f"entire model flops: {flops}, params: {params}")
    module_flops = []
    module_param = []
    for i in range(len(modules)):
        module = modules[i]
        flops, params = thop.profile(module, inputs=(inputs,))
        module_flops.append(flops)
        module_param.append(params)
        print(f"module_{i} flops: {flops}, params: {params}")

    compose_flops =  sum(module_flops)
    compose_params = sum(module_param)

    with open(f'{module_dir}/analaysis/flops.txt', 'w', newline='', encoding='utf-8')as f:
        f.write(f'entire model flops: {model_flops},params:{model_params}\n')
        for i in range(len(module_flops)):
            f.write(f"module_{i} flops: {module_flops[i]}\n")
        f.write (f"Composed model FLOPs: {compose_flops},params:{compose_params}\n")

def evaluate_module_per_class(modules,test_data_loaders):
    result=[]
    for module in modules:
        tmp = []
        for i in range(10):
            module.eval()
            module.to(device)
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_data_loaders[i]:
                    images, labels = images.to(device), labels.to(device)
                    outputs = module(images)
                    predicted = torch.argmax(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            tmp.append(correct / total)
        result.append(tmp)
    return result

def fine_tune_module(modules, train_loaders, lr=0.005, epochs=10):
    num_modules= len(modules)
    for i in range(num_modules):
        module = modules[i]
        train_loader = train_loaders[i]
        module.to(device)
        param_to_train = [param for name,param in  module.head.named_parameters()]
        optimizer = SGD(param_to_train, lr=lr, weight_decay=0.5, momentum=0.9, nesterov=True)
        criterion = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs)):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = module(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            


def test_module(modules,test_data_loader):
    for model in modules:
        model.eval()
        model.to(device)

    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in test_data_loader:
           
            votes = torch.zeros(images.size(0), len(modules))  
            images, labels = images.to(device), labels.to(device)
            
            for i, model in enumerate(modules):
                outputs = model(images)  
                # _, predicted = torch.max(outputs.data, 1)  
                # print(i,outputs[0][1])
                votes[:, i] = outputs[:,1]  
            
            final_predictions = []
           
            final_predictions = torch.argmax(votes, dim=1)
            final_predictions = torch.tensor(final_predictions,device=device)

          
            total += labels.size(0)
            correct += (final_predictions == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")
def main():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if dataset == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == 'kmnist':
        train_dataset = datasets.KMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    modules = load_modules(module_dir,10,f'{model_name}')
    origin_model = load_model(f'{model_name}')
    
    # param = count_nonzero_weights(origin_model)

    # m_param = 0
    # for module in modules:
    #     m_param += count_nonzero_weights(module)
    # print(f"origin model params: {param}")
    # print(f"module params: {m_param}")
    # print(f"module params ratio: {m_param/param}")
    # origin_model.load_state_dict(torch.load(model_dir))
    # # calculate_jaccard_similarity(modules)
    calculate_flops(test_dataset,modules,origin_model)

    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # # test_module(modules,test_dataloader)

    # labelwise_dataloaders_test = create_labelwise_dataloaders(test_dataset, batch_size=batch_size)
    # labelwise_dataloaders_train = create_labelwise_dataloaders(train_dataset, batch_size=batch_size)
    # result = evaluate_module_per_class(modules,labelwise_dataloaders_test)
    # with open(f'{module_dir}/analaysis/accuracy.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(result)

    # random_modules = generate_random_modules(modules,origin_model)
    # fine_tune_module(random_modules,labelwise_dataloaders_train)
    # test_module(random_modules,test_dataloader)
    # random_result = evaluate_module_per_class(random_modules,labelwise_dataloaders_test)
    # with open(f'{module_dir}/analaysis/accuracy_random.csv', 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(random_result)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', choices=['fc3','fc5'],required=True)
    parser.add_argument('--dataset', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--lr_head', type=float, default=0.005)
    parser.add_argument('--lr_modularity', type=float, default=0.5)
    parser.add_argument('--epoch', type=str, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    print(args)
    root_dir = '/bdata/rq/modularization/decomposeWithMasks'
    data_dir =root_dir + '/data'
    model_dir = f'{root_dir}/models/{args.model}_{args.dataset}.pth'
    module_dir = f'{root_dir}/models/{args.model}_{args.dataset}'
    
    model_name = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    main()