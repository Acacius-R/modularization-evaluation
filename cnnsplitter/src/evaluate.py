from utils.configure_loader import load_configure
from utils.model_loader import load_model
from utils.module_tools import *
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.dataset_loader import get_dataset_loader
import json
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import ttest_rel
import thop

def evaluate_modules_per_class(modules, dataset, num_classes):

    results=[]

    # 遍历每个模块
    for module_idx, (module,_) in enumerate(tqdm(modules, desc="Evaluating modules")):
        # print(f"module_idx: {module_idx}")
        # module_results = {}
        all_preds = []
        all_labels = []
        # with torch.no_grad():
        #     for inputs, labels in dataset:
        #         inputs = inputs.to(device)
        #         outputs = module(inputs)
        #         # print(f"outputs: {outputs.shape}")
        #         preds = torch.argmax(outputs, dim=1).cpu().numpy()
        #         all_preds.extend(preds)
        #         all_labels.extend(labels.cpu().numpy())
        tmp = []
        outputs,all_labels = module_predict(module, dataset)
        all_preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        # print(all_preds.shape,all_labels.shape)
        class_correct=[0]*num_classes
        class_total=[0]*num_classes
        correct = (all_preds == all_labels).squeeze()
        for i in range(len(all_labels)):
            label = all_labels[i]
            class_correct[label] += correct[i]
            class_total[label] += 1

        tmp = []
        for i in range(num_classes):
            tmp.append(class_correct[i] / class_total[i])
            # print(f"tmp: {tmp}")
        results.append(tmp)

    return results

def compare_with_random(n):
    random_results = []
    for i in  range(n):
        modules = load_modules(configs, randomseed==randomseed)
        acc = evaluate_ensemble_modules(modules, test_dataset)
        random_results.append(acc)
    
    acc = evaluate_ensemble_modules(modules, test_dataset)
    t_statistic, p_value = ttest_rel([acc]*n, random_results)

def calculate_flops():
    inputs, labels = next(iter(test_dataset))
    inputs = inputs.to(device)
    flops, params = thop.profile(model, inputs=(inputs,))
    print(f"entire model flops: {flops}, params: {params}")
    module_flops = []
    module_param = []
    for i, (module,_) in enumerate(modules):
        flops, params = thop.profile(module, inputs=(inputs,))
        module_flops.append(flops)
        module_param.append(params)
        print(f"module_{i} flops: {flops}, params: {params}")

    flops_gap =  sum(module_flops)
    print(f"FLOPs  Modules: {flops_gap},params:{sum(module_param)}")

    # with open(f'{configs.workspace}/flops.txt', 'w', newline='', encoding='utf-8')as f:
    #     f.write(f'entire model flops: {flops}\n')
    #     for i in range(len(module_flops)):
    #         f.write(f"module_{i} flops: {module_flops[i]}\n")
    #     f.write (f"FLOPs gap ( Modules - module): {flops_gap}")
#测试代码
def compare_model_parameters(model_a, model_b):
    """
    Compares the parameters of two models and returns True if they are identical.

    Args:
        model_a (torch.nn.Module): First model
        model_b (torch.nn.Module): Second model

    Returns:
        bool: True if all parameters are identical, False otherwise.
    """
    params_a = model_a.state_dict()
    params_b = model_b.state_dict()

    if params_a.keys() != params_b.keys():
        return False

    for key in params_a.keys():
        if not torch.equal(params_a[key], params_b[key]):
            return False

    return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn','cifar100'])
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    configs = load_configure(model_name,dataset_name)
    # print(configs.num_classes)
    randomseed = 302
    modules,model=load_modules1(configs,return_trained_model=True,randomseed=randomseed)
    #modules = load_modules(configs)
    composed_para = 0
    #不同模块是不一样的
    # print(compare_model_parameters(modules[0][0],modules[1][0]))    
    # for module in modules:
    #     total_params = sum(p.numel() for p in module[0].parameters() if p.requires_grad)
    #     # nonzero_params = sum(torch.count_nonzero(p).item() for p in module[0].parameters() if p.requires_grad)
    #     # print(f"Total Params: {total_params}")
    #     # # print(f"Module Non-zero Params: {nonzero_params}")
    #     # print("-"*80)
    #     composed_para += total_params
    # print(f"Composed Model Params: {composed_para}")
    
    #total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)
    # print(f"Total Params: {total_params}")
    # print(f"Model Non-zero Params: {nonzero_params}")

    # print(len(modules))
    load_dataset = get_dataset_loader(dataset_name)
    test_dataset = load_dataset(configs.dataset_dir, is_train=False, batch_size=128,
                                num_workers=1, pin_memory=True)
    # # print (modules[0])
    calculate_flops()    

    # results = evaluate_modules_per_class(modules, test_dataset, 10)
    # print(results)
    # with open(f'{configs.workspace}/accuracy_random.csv', 'w', newline='', encoding='utf-8')as f:
    #     writer = csv.writer(f)
    #     writer.writerows(results)

    # output_json_path='result.json'
    # with open(output_json_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    #     print(f"Results saved to {output_json_path}")
    # results_random = load_results_from_json('random_result.json')
    # results_best = load_results_from_json('result.json')
    # output_image_path = "accuracy_gap_comparison2.png"  # 指定保存路径
    # plot_accuracy_comparison(results_random, results_best, output_image_path)

    # acc = evaluate_ensemble_modules(modules, test_dataset)
    # print(f"Ensemble model accuracy: {acc:.4f}")

    # compare_with_random(10)
    # ratio = calculate_module_param_ratio(modules, model)
    # print("param rate",np.mean(ratio))
    
    # similariyt = calculate_all_jaccard_indices(modules)
    # with open(f'{configs.workspace}/similariyt.csv', 'w', newline='', encoding='utf-8')as f:
    #     writer = csv.writer(f)
    #     writer.writerows(similariyt)