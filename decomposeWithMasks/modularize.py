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
import torch.nn.functional as F
def stemming_loss(current_scores, previous_scores_list, alpha=0.05, p=1):
    """
    """
    
    

    regularization_loss = 0.0
    L = len(current_scores)
    M = next(iter(current_scores.values())).shape[0]
    for name,scores in current_scores.items():
        
        layer_scores = torch.stack([prev_scores[name] for prev_scores in previous_scores_list])  # (N, M, M)
        avg_S = layer_scores.mean(dim=0) 
        
        diff = scores - avg_S
        regularization_loss += torch.norm(diff, p=p)
    
    normalization_factor = L * (M ** 2)
    regularization_loss = (alpha / normalization_factor) * regularization_loss
    return regularization_loss

def modularize(model,data_loaders,threshold,epochs):
    num_modules = len(data_loaders)
    mask_models = []
    for i in range(num_modules):
        maskedmodel = MaskModel(model,threshold,f'{model_name}')
        maskedmodel.to(device)
        mask_models.append(maskedmodel) 
    
    # criterion = nn.CrossEntropyLoss()
    previous_scores_list = []
    first =True
    train_head = False
    previous_scores_list=[]
    for epoch in tqdm(range(epochs)):
        temp = []
        if epoch >int(0.9*epoch):
            train_head = True
        for i in range(num_modules):
            # print(i)
            dataloader = data_loaders[i]
            module = mask_models[i]
            module.to(device)
            
            if train_head:
                param_to_train = [param for name,param in  module.masked_model.head.named_parameters()]
                learning_rate = lr_head
            else:
                param_to_train = [param for name, param in module.scores.items()]
                learning_rate = lr_modularity
            # print(param_to_train)
            optimizer = SGD(param_to_train, lr=learning_rate, weight_decay=0.5, momentum=0.9, nesterov=True)
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = module(images)
                # print(outputs,labels)
                loss = F.cross_entropy(outputs, labels)
                if not first and not train_head:
                    loss += stemming_loss(maskedmodel.scores, previous_scores_list, alpha=args.alpha)
                loss.backward()
                optimizer.step()
            temp.append(module.scores)
            module.apply_mask()
        previous_scores_list=temp
        train_head =False
    for i in range(num_modules):
        torch.save(mask_models[i].masked_model.state_dict(),f'{save_dir}/modules_{i}.pth')
    return mask_models

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
                votes[:, i] = outputs[:,1]  

            
            final_predictions = []
            for i in range(votes.size(0)):  
                # vote_count = torch.bincount(votes[i].long())  
                final_predictions.append(votes[i].argmax().item())  

            final_predictions = torch.tensor(final_predictions,device=device)

            
            total += labels.size(0)
            correct += (final_predictions == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")
def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # load dataset
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    labelwise_dataloaders = create_labelwise_dataloaders(train_dataset, batch_size=batch_size)
    model = load_model(f'{model_name}')
    model.load_state_dict(torch.load(model_dir))

    
    threshold = 0.5
    modules = modularize(model,labelwise_dataloaders,threshold,epochs)
    # modules = load_modules(save_dir,10,f'{model_name}_{dataset}')
    test_module(modules,test_loader)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fc3','fc5'],required=True)
    parser.add_argument('--dataset', choices=['mnist', 'fmnist', 'kmnist'])
    parser.add_argument('--lr_head', type=float, default=0.005)
    parser.add_argument('--lr_modularity', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()
    print(args)
    root_dir = '/bdata/rq/modularization/decomposeWithMasks'
    data_dir =root_dir + '/data'
    model_dir = root_dir + f'/models/{args.model}_{args.dataset}.pth'
    model_name = args.model
    dataset = args.dataset
    lr_modularity = args.lr_modularity
    lr_head = args.lr_head
    batch_size = args.batch_size
    epochs = args.epoch
    save_dir = f'models/{model_name}_{dataset}'
    check_dir(save_dir)
    main()