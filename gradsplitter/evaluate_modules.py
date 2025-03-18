import argparse
import sys
import torch
sys.path.append('')
sys.path.append('..')
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader
from tqdm import tqdm
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy.stats import ttest_rel
import numpy as np
import csv
import thop 
def evaluate_ensemble_modules(modules, dataset):
    modules_outputs = []
    data_labels = None
    for each_module in modules:
        outputs, labels = module_predict(each_module, dataset)
        # print(f"outputs: {outputs[0]}")
        modules_outputs.append(outputs)
        if data_labels is None:
            data_labels = labels
        # else:
            # assert (data_labels == labels).all()
    modules_outputs = torch.stack(modules_outputs, dim=1)
    # print(f"modules_outputs: {modules_outputs[0]}")
    final_pred = torch.argmax(modules_outputs, dim=1).squeeze()
    # print(f"final_pred: {final_pred[0]}")
    # print(f"final_pred.shape: {final_pred.shape}, data_labels.shape: {data_labels.shape}")
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))
    return acc.item()

def calculate_flops(model,modules,module_eval_dataset):
    inputs, labels = next(iter(module_eval_dataset))
    inputs = inputs.to(device)
    entire_flops, entire_params = thop.profile(model, inputs=(inputs,))
    
    module_flops = []
    module_param=[]

    for i,module in enumerate(modules):
        flops, params = thop.profile(module, inputs=(inputs,))
        module_flops.append(flops)
        module_param.append(params)
        # print(f"module_{i} flops: {flops}, params: {params}")

    flops_gap =  sum(module_flops)
    print(f"entire model flops: {entire_flops}, params: {entire_params}")
    print(f"FLOPs {flops_gap},param:{sum(module_param)}")

    with open(f'{configs.workspace}/flops.txt', 'w', newline='', encoding='utf-8')as f:
        f.write(f'entire model flops: {flops}\n')
        for i in range(len(module_flops)):
            f.write(f"module_{i} flops: {module_flops[i]}\n")
        f.write (f"FLOPs gap ( Modules - module): {flops_gap},")

def evaluate_module_f1(module, dataset, target_class):
    outputs, labels = module_predict(module, dataset)
    predicts = (outputs > 0.5).int().squeeze(-1)
    labels = (labels == target_class).int()
    acc = torch.div(torch.sum(predicts == labels), len(labels))
    precision = torch.sum(predicts * labels) / torch.sum(predicts)
    recall = torch.sum(predicts * labels) / torch.sum(labels)
    f1 = 2 * (precision * recall) / (precision + recall)
    return acc.item()


def get_module_weights(module):
    weights = []
    for param in module.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)

def jaccard_similarity(list1, list2):
    intersection = len(np.intersect1d(list1, list2))
    union = len(np.union1d(list1, list2))
    return float(intersection) / union

def calculate_jaccard_similarity(modules):
    print("Calculating Jaccard Similarity of Weights")
    print('-' * 80)
    num_modules = len(modules)
    weights_to_compare = [get_module_weights(module) for module in modules]
    
    result = []
    for i in range(num_modules):
        tmp = []
        for j in range(num_modules):
            similarity = jaccard_similarity(weights_to_compare[i], weights_to_compare[j])
            tmp.append(similarity)
            # print(f"Module {i} and Module {j} Jaccard similarity of weights: {similarity}")
        result.append(tmp)

    with open(f'{configs.workspace}/similarity.csv', 'w', newline='', encoding='utf-8')as f:
        writer = csv.writer(f)
        writer.writerows(result)
    
    return result

def main():
    estimator_idx = args.estimator_idx
    print(f'Estimator {estimator_idx}')
    print('-' * 80)
    global random_seed
    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)
    print(configs.trained_model_path)
    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    module_path = configs.best_module_path
    module_eval_dataset, _ = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None)
    modules=[]
    random_module=[]
    composed_para =0
    # evaluate each module
    for i in range(configs.num_classes):
        print(f'Loading Module {i}')
        module = load_module(module_path, trained_model, i,random_seed=None)
        modules.append(module)
    # for module in modules:
    #     total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    #     # nonzero_params = sum(torch.count_nonzero(p).item() for p in module[0].parameters() if p.requires_grad)
    #     # print(f"Total Params: {total_params}")
    #     # # print(f"Module Non-zero Params: {nonzero_params}")
    #     # print("-"*80)
    #     composed_para += total_params
    # print(f"Composed Model Params: {composed_para}")
        # result = evaluate_module_f1(module, module_eval_dataset, i)
        # print(f'Module {i} acc:{result:.4f}')
        # module = load_module(module_path, trained_model, i,random_seed)
        # random_module.append(module)
        # result = evaluate_module_f1(module, module_eval_dataset, i)
        # print(f'Module random {i} acc:{result:.4f}')

    model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total Params: {total_params}")
    
    #RQ3
    calculate_flops(model,modules,module_eval_dataset)
    # cross_acc = []

    #RQ1&RQ2
    # for module in tqdm(modules):
    #     tmp = []
    #     for i in range(configs.num_classes):
    #         tmp.append(evaluate_module_f1(module, module_eval_dataset, i))
    #     cross_acc.append(tmp)
    # with open(f'{configs.workspace}/accuracy.csv', 'w', newline='', encoding='utf-8')as f:
    #     writer = csv.writer(f)
    #     writer.writerows(cross_acc)
    # calculate_jaccard_similarity(modules)

    #RQ4
    # acc = evaluate_ensemble_modules(modules,module_eval_dataset)
    # random_acc = evaluate_ensemble_modules(random_module,module_eval_dataset)
    # print(f'Ensemble acc:{acc:.4f}, Random Ensemble acc:{random_acc:.4f}')
    # random_acc = []
    # n=10
    
    # print("Evaluating Random Modules")
    # print('-' * 80)
    # for i in range(n):
    #     random_module = []
    #     for j in range(configs.num_classes):
    #         module = load_module(module_path, trained_model, j,random_seed)
    #         random_module.append(module)
    #     random_acc.append(evaluate_ensemble_modules(random_module,module_eval_dataset))
    #     random_seed+=1

    # t_statistic, p_value = ttest_rel([acc]*n, random_acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['vgg16', 'rescnn', 'simcnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn','cifar100'])
    parser.add_argument('--estimator_idx', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=45)
    args = parser.parse_args()
    print(args)
    random_seed = 303
    model_name = args.model
    dataset_name = args.dataset
    lr_head = args.lr_head
    batch_size = args.batch_size
    estimator_idx = args.estimator_idx
    epochs = args.epochs
    configs = load_configure(model_name, dataset_name)
    configs.set_estimator_idx(estimator_idx)
    dataset_dir = configs.dataset_dir
    print(dataset_dir)
    load_dataset = get_dataset_loader(dataset_name, for_modular=True)
    train_dataset, val_dataset = load_dataset(dataset_dir, is_train=True, split_train_set='8:2',
                                                                  shuffle_seed=estimator_idx, is_random=False,
                                                                  batch_size=batch_size, num_workers=1, pin_memory=True)
    print(args)
    main()