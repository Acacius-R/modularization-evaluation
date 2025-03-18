import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
from models.fc3 import FC3
from models.fc5 import FC5
def train(model,epochs,train_data_loader,val_data_loader,save_dir):
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        for images, labels in tqdm(train_data_loader):
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step() 
            running_loss += loss.item()
        
        
        accuracy = test(model,val_data_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_dir)
            print(f"New best model saved with accuracy: {100 *accuracy:.2f}% in {save_dir}" )


def test(model,test_data_loader):
    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():  
        for images, labels in test_data_loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # print(f"Test Accuracy: {100 * correct / total}%")
    return correct / total

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    root_dir = '/bdata/rq/modularization/decomposeWithMasks'
    data_dir =root_dir + '/data'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fc3','fc5'],required=True)
    parser.add_argument('--dataset', choices=['mnist', 'fmnist', 'kmnist'],required=True)
    parser.add_argument('--epoch', type=str, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    dataset_name = args.dataset
    model_name =args.model
    epochs = args.epoch
    batch_size = args.batch_size
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'fmnist':
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'kmnist':
        train_dataset = datasets.KMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    save_dir = f'{root_dir}/models/{args.model}_{dataset_name}.pth'
    if model_name == 'fc3':
        model = FC3()
    elif model_name == 'fc5':
        model = FC5()
    else:
        raise ValueError("Unsupported model")
    train(model,epochs,train_loader,test_loader,save_dir)
    model.load_state_dict(torch.load(save_dir))
    acc = test(model,test_loader)
    print(f"Final test accuracy: {100*acc:.2f}%")