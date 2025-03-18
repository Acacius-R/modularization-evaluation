import argparse
import copy
import torch
from tqdm import tqdm
from utils.checker import check_dir
from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.model_loader import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.module_tools import load_modules, evaluate_ensemble_modules

def main():
    print(f'Using {device}')
    configs = load_configure(model_name, dataset_name)
    dataset_dir = configs.dataset_dir
    check_dir(configs.trained_model_dir)
    save_path = f'{configs.trained_model_dir}/{configs.trained_entire_model_name}'
    print(configs.best_generation)
    load_dataset = get_dataset_loader(configs.dataset_name)
    test_dataset = load_dataset(configs.dataset_dir, is_train=False, batch_size=128,
                                num_workers=1, pin_memory=True)
    print(configs.best_generation)
    modules = load_modules(configs)
    print(type(modules))
    print(type(modules[0]))
    print(len(modules[0]))
    #modules = [m[0] for m in modules]
    print(type(modules))
    test_acc = evaluate_ensemble_modules(modules, test_dataset)
    print(test_acc)
    
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
