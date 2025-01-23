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


def evaluate_module_f1(module, dataset, target_class):
    outputs, labels = module_predict(module, dataset)
    predicts = (outputs > 0.5).int().squeeze(-1)
    labels = (labels == target_class).int()
    acc = torch.div(torch.sum(predicts == labels), len(labels))
    precision = torch.sum(predicts * labels) / torch.sum(predicts)
    recall = torch.sum(predicts * labels) / torch.sum(labels)
    f1 = 2 * (precision * recall) / (precision + recall)
    return acc

def retrain_head(module, train_dataset,lr_head,epochs,target_class,val_dataset):
    module.train()
    for name, param in module.module_head.named_parameters():
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    total_samples = len(train_dataset)
    class_1_count = 0
    for batch_inputs, batch_labels in train_dataset:
        class_1_count += sum((label == target_class).int() for label in batch_labels)

    class_0_count = total_samples - class_1_count

    class_0_weight = class_1_count / total_samples
    class_1_weight = class_0_count / total_samples
    weights = torch.tensor([class_0_weight, class_1_weight]).to(device)
    head_param = [param for param in module.module_head.parameters()]
    optimizer = torch.optim.Adam(head_param, lr=lr_head)
    best_acc = 0.0
    best_model_wts = None
    for epoch in range(epochs):
        # print(f'epoch {epoch}')
        for batch_inputs, batch_labels in tqdm(train_dataset, ncols=100, desc='train'):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            batch_inputs.requires_grad = True
            outputs = module(batch_inputs).squeeze(-1)
            # print(f"outputs: {outputs}")
            # print(f"outputs.grad_fn: {outputs.grad_fn}")
            
            optimizer.zero_grad()
            # print('outputs.shape:',outputs.shape, 'batch_labels.shape:',batch_labels.shape)
            # print('outputs:',outputs)
            # print('batch_labels:',batch_labels)
            
            batch_labels = (batch_labels == target_class).float()
            # print(f"class_1_sum: {class_1_sum}, class_0_sum: {class_0_sum}")
            loss = F.binary_cross_entropy_with_logits(outputs, batch_labels, weight=weights[batch_labels.long()])
            # print(f"loss: {loss.item()}")
            loss.backward()
            # for name, param in module.module_head.named_parameters():
            #     print(f"{name} grad: {param.grad}")
            optimizer.step()

        module.eval()
        val_acc = evaluate_module_f1(module, val_dataset, target_class)
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = module.state_dict()
        module.train()

    if best_model_wts is not None:
        module.load_state_dict(best_model_wts)
            
        # print(f"acc : {sum(epoch_train_acc) / len(epoch_train_acc) * 100:.2f}%\n")
    module.eval()

def main():
    estimator_idx = args.estimator_idx
    print(f'Estimator {estimator_idx}')
    print('-' * 80)

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)

    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    module_path = configs.best_module_path

    # evaluate each module
    for i in range(configs.num_classes):
        module_eval_dataset, _ = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None)
        module = load_module(module_path, trained_model, i)
        # print(f"Before reset_module_head_weights for class {i}:")
        # for name, param in module.module_head.named_parameters():
        #     print(f"{name}: {param.data}")
        # module.reset_module_head_weights()
        # print(f"After reset_module_head_weights for class {i}:")
        # for name, param in module.module_head.named_parameters():
        #     print(f"{name}: {param.data}")

        # result = evaluate_module_f1(module, module_eval_dataset, i)
        # print(f'acc_b:{result:.4f}')
        retrain_head(module, train_dataset, lr_head, epochs,i,val_dataset)
        # print(f"After retrain head for class {i}:")
        # for name, param in module.module_head.named_parameters():
        #     print(f"{name}: {param.data}")
        # module_eval_dataset, _ = load_dataset(dataset_dir, is_train=False, shuffle_seed=None, is_random=None)
        result = evaluate_module_f1(module, module_eval_dataset, i)
        print(f'acc:{result:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'rescnn', 'incecnn'])
    parser.add_argument('--dataset', choices=['cifar10', 'svhn'])
    parser.add_argument('--estimator_idx', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_head', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=45)
    args = parser.parse_args()
    print(args)

    model_name = args.model
    dataset_name = args.dataset
    lr_head = args.lr_head
    batch_size = args.batch_size
    estimator_idx = args.estimator_idx
    epochs = args.epochs
    configs = load_configure(model_name, dataset_name)
    configs.set_estimator_idx(estimator_idx)
    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(dataset_name, for_modular=True)
    train_dataset, val_dataset = load_dataset(dataset_dir, is_train=True, split_train_set='8:2',
                                                                  shuffle_seed=estimator_idx, is_random=False,
                                                                  batch_size=batch_size, num_workers=1, pin_memory=True)
    print(args)
    main()