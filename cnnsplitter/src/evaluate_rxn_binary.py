import argparse
import copy
import torch
from tqdm import tqdm
from utils.checker import check_dir
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.model_loader import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.module_tools import load_modules, evaluate_ensemble_modules,load_modules1 ,load_modules2


def test(model, test_loader,i):
    epoch_acc = []
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            pred = torch.argmax(outputs, dim=1)
            binary_pred = (pred == i).int()
            for j in range(10):
                binary_labels = (batch_labels==j).int()
                correct = (binary_pred == binary_pred).squeeze()
                class_correct[j]+=correct
                class_total[j]+=len(batch_labels)
    ans = [i/j for i,j in zip(class_correct,class_total) ]
    print(ans)
    '''
    for i in range(10):
        if class_total[i] > 0:
            print(f"Accuracy of class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Accuracy of class {i}: N/A (no samples)")'
    '''
# test with sigle clss
def test_split(model, test_loader,class_i):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            pred = torch.argmax(outputs, dim=1)

            # get all labels and preds
            all_preds.append(pred)
            all_labels.append(batch_labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    binary_preds = (all_preds==class_i).int()

    #treat as a binary classification problem
    ans = []
    for j in range(10):
        binary_labels = (all_labels==j).int()
        ones_idx = torch.where(binary_labels == 1)[0]
        zeros_idx = torch.where(binary_labels == 0)[0]
        selected_zeros_idx = zeros_idx[torch.randperm(len(zeros_idx))[:1000]]  # 随机采样 1000 个
        # merge index
        selected_indices = torch.cat((ones_idx, selected_zeros_idx))  
        binary_labels1 = binary_labels[selected_indices]
        binary_preds1 = binary_preds[selected_indices]      
        accuracy = (binary_preds1 == binary_labels1).float().mean().item()
        ans.append(accuracy)
    
    print(ans) 
    #print(all_preds.shape)
    #print(all_labels.shape)
    '''
            binary_pred = (pred == class_i).int()
            binary_labels = (batch_labels == class_i).int()

            # 计算 correct
            correct1 = (binary_pred == binary_labels).squeeze()
            #correct = (pred == batch_labels).squeeze()
            #记录每个类别的准确率
            correct += sum(correct1)
            all += len(batch_labels)
    print(correct/all)        
    '''

def main():
    print(f'Using {device}')
    configs = load_configure(model_name, dataset_name)
    dataset_dir = configs.dataset_dir
    check_dir(configs.trained_model_dir)
    save_path = f'{configs.trained_model_dir}/{configs.trained_entire_model_name}'
    print(configs.best_generation)
    load_dataset = get_dataset_loader(configs.dataset_name)
    test_dataset = load_dataset(configs.dataset_dir, is_train=False, batch_size=128,
                                num_workers=1, pin_memory=True,is_random=False)
    print(configs.best_generation)

    modules = load_modules(configs)
    modules = load_modules2(configs,randomseed=103) #laod random modules
    
    test_acc = evaluate_ensemble_modules(modules, test_dataset)
    print(test_acc)
   
    test_loader = load_dataset(dataset_dir, is_train=False,
                               batch_size=batch_size, num_workers=1, pin_memory=True)

    for i,m in enumerate(modules):
        model,_ = m
        model.eval()
        test_split(model, test_loader,i)
        print('---------------')
        #test_split(model,test_loader,i)
        #print('---------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn', 'svhn_5','cifar100'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--schedule', type=str, default='60,120', help='Decrease learning rate at these epochs.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    args = parser.parse_args()
    print(args)
    print()

    model_name = args.model
    dataset_name = args.dataset
    lr = args.lr
    batch_size = args.batch_size
    gamma = args.gamma
    lr_schedule = [int(s) for s in args.schedule.split(',')]
    n_epochs = args.epochs
    early_stop = args.early_stop
    evaluation = args.evaluation
    main()
