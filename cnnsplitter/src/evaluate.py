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
def load_results_from_json(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results
def plot_accuracy_comparison(results_random, results_best, output_image_path=None):
    """
    绘制两种选择中不同模块在不同类别数据上的准确率对比图。

    参数:
        results_random: 字典，随机选择模块的评估结果。
        results_best: 字典，最佳选择模块的评估结果。
        output_image_path: 字符串，保存图像的路径。如果为 None，则显示图像。
    """
    num_modules = len(results_random)
    num_classes = len(results_random['module_0'])

    # 提取准确率数据
    random_accuracies = np.zeros((num_modules, num_classes))
    best_accuracies = np.zeros((num_modules, num_classes))

    for module_idx in range(num_modules):
        for class_idx in range(num_classes):
            if class_idx != module_idx:
                continue
            random_accuracies[module_idx, class_idx] = results_random[f'module_{module_idx}'][f'class_{class_idx}']['accuracy']
            best_accuracies[module_idx, class_idx] = results_best[f'module_{module_idx}'][f'class_{class_idx}']['accuracy']

    # 计算准确率差距
    accuracy_gaps = best_accuracies - random_accuracies

    # 设置绘图参数
    x = np.arange(num_classes)  # 类别索引
    width = 0.35  # 柱状图宽度

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    for module_idx in range(num_modules):
        plt.bar(x + module_idx * width, accuracy_gaps[module_idx], width, label=f'Module {module_idx}')

    # 添加标题和标签
    plt.title('Accuracy Gap (Best - Random) by Module and Class')
    plt.xlabel('Class')
    plt.ylabel('Accuracy Gap')
    plt.xticks(x + width * (num_modules - 1) / 2, [f'{i}' for i in range(num_classes)])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存或显示图像
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight')
        print(f"Plot saved to {output_image_path}")
    else:
        plt.show()

def evaluate_modules_per_class(modules, dataset, num_classes):
    """
    评估不同模块在不同标签数据上的表现。

    参数:
        modules: 列表，包含多个模块（通常是神经网络模型）。
        dataset: 数据集对象（通常是 DataLoader），包含输入数据和标签。
        num_classes: 整数，表示类别数量。

    返回值:
        results: 字典，包含每个模块在每个类别上的评估指标。
                格式: {
                    'module_0': {
                        'class_0': {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...},
                        'class_1': {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...},
                        ...
                    },
                    'module_1': { ... },
                    ...
                }
    """
    results = {}

    # 遍历每个模块
    for module_idx, (module,_) in enumerate(tqdm(modules, desc="Evaluating modules")):
        module_results = {}
        all_preds = []
        all_labels = []

        # 使用模块对数据集进行预测
        with torch.no_grad():
            for inputs, labels in dataset:
                inputs = inputs.to(device)
                outputs = module(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # 计算每个类别的评估指标
        for class_idx in range(num_classes):
            # 获取当前类别的预测和标签
            class_preds = [1 if p == class_idx else 0 for p in all_preds]
            class_labels = [1 if l == class_idx else 0 for l in all_labels]

            # 计算指标
            accuracy = accuracy_score(class_labels, class_preds)
            precision = precision_score(class_labels, class_preds, zero_division=0)
            recall = recall_score(class_labels, class_preds, zero_division=0)
            f1 = f1_score(class_labels, class_preds, zero_division=0)

            # 存储结果
            module_results[f'class_{class_idx}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # 将当前模块的结果存入总结果
        results[f'module_{module_idx}'] = module_results

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn_5'])
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    configs = load_configure(model_name,dataset_name)
    modules,model=load_modules(configs,return_trained_model=True)
    load_dataset = get_dataset_loader(dataset_name)
    test_dataset = load_dataset(configs.dataset_dir, is_train=False, batch_size=128,
                                num_workers=1, pin_memory=True)
    # print (modules[0])

    # results = evaluate_modules_per_class(modules, test_dataset, 10)

    # output_json_path='random_result.json'
    # with open(output_json_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    #     print(f"Results saved to {output_json_path}")
    results_random = load_results_from_json('random_result.json')
    results_best = load_results_from_json('result.json')
    output_image_path = "accuracy_gap_comparison2.png"  # 指定保存路径
    plot_accuracy_comparison(results_random, results_best, output_image_path)
    