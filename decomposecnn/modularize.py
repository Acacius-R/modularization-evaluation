import argparse
from ast import mod
import copy
import torch
from tqdm import tqdm
from utils.checker import check_dir
from utils.load_datasets import get_dataset_loader
from utils.model_loader import load_model
from concern_identification import CI
from concern_modularization import back_track
from utils.tools import extract_weights
from tqdm import tqdm


def modularize(model,target_class,inputs):
    module = model
    layer_info = module.layer_info
    convW, convB, denseW, denseB = extract_weights(model)
    # print(layer_info[0])
    update_weights = []
    update_biases = []
    for layer in layer_info:
        if layer['params']['weight'] is not None:
            update_weights.append(torch.zeros_like(layer['params']['weight']))
        else:
            update_weights.append(None)

        if layer['params']['bias'] is not None:
            update_biases.append(torch.zeros_like(layer['params']['bias']))
        else:
            update_biases.append(None)

    
    first = True
    for input in tqdm(inputs):
        conv_maps = CI(model,input,update_weights[-1],update_biases[-1],first)
        first = False
        update_weights,update_biases,conv_maps = back_track(model,target_class,update_weights,update_biases,conv_maps,input)
    
    conv_idx = 0
    for layer_idx, layer in enumerate(module.layers):
        if isinstance(layer, torch.nn.Conv2d):
            # 对于卷积层，根据 conv_map 更新权重
            if hasattr(layer, 'weight'):
                weight = convW[conv_idx]
                bias = convB[conv_idx]
                inactive_nodes = conv_maps[conv_idx]
                # 将不需要的权重设置为 0
                for node in inactive_nodes:
                    weight[:, node] = 0
                layer.weight.data = weight
                layer.bias.data = bias
                conv_idx += 1
        else:
            # 对于其他层，直接更新权重
            if hasattr(layer, 'weight'):
                layer.weight.data = update_weights[layer_idx]
                layer.bias.data = update_biases[layer_idx]
    
    return module

def main():
    save_path = f'./models/{model_name}_{dataset_name}'
    model = load_model(model_name,num_class)
    model.load_state_dict(torch.load(f'/bdata/rq/modularization/decomposecnn/models/{model_name}_{dataset_name}.pth'))
    load_dataset = get_dataset_loader(dataset_name)
    model.to(device)
    train_loader, val_loader = load_dataset(dataset_dir, is_train=True,
                                            batch_size=batch_size, num_workers=1, pin_memory=True)
    for j in range(num_class):
        print(f"Module {j} in progress....")
        print("-"*50)
        inputs = []
        for data, target in train_loader:
            if target.item() == j:
                inputs.append(data.to(device))


        module = modularize(model, j, inputs)
        torch.save(module.state_dict(), f'{save_path}/module_{j}.pth')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet20'])
    parser.add_argument('--dataset', choices=['cifar10'])
    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    dataset_dir = '/bdata/rq/modularization/decomposecnn/datasets'
    batch_size = 1
    num_class = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()