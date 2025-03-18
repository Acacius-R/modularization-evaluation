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
import torch.nn.functional as F

def evaluate_modules_per_class(modules, dataset, num_classes):

    module_outputs =[[] for _ in range(len(modules))]
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
        
        outputs,all_labels = module_predict(module, dataset)
        outputs = F.softmax(outputs, dim=1)
        class_above_threshold = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        # all_preds = torch.argmax(outputs, dim=1).cpu().numpy()
        class_outputs = {i: [] for i in range(num_classes)}
        for i in range(len(all_labels)):
            label = all_labels[i].item()
            output_value = outputs[i,module_idx].cpu().numpy().item()
            class_outputs[label].append(output_value)
            class_total[label] += 1
            if output_value > 0.5:
                class_above_threshold[label] += 1

        for class_idx in range(num_classes):
            if class_total[class_idx] > 0:
                proportion_above_threshold = class_above_threshold[class_idx] / class_total[class_idx]
            else:
                proportion_above_threshold = 0
            print(f"Module {module_idx}, Class {class_idx}:")
            print(f"Proportion of outputs > 0.5: {proportion_above_threshold:.2f}")
            print(f"Outputs: {class_outputs[class_idx][:10]}")
        with open(f'{configs.workspace}/module_outputs_{module_idx}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for class_idx in range(num_classes):
                    writer.writerow([class_outputs[class_idx]])
        



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
    modules,model=load_modules1(configs,return_trained_model=True,randomseed=None)
    load_dataset = get_dataset_loader(dataset_name)
    
    test_dataset = load_dataset(configs.dataset_dir, is_train=False, batch_size=128,
                                num_workers=1, pin_memory=True)
    evaluate_modules_per_class(modules, test_dataset, configs.num_classes)