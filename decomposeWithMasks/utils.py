from models.fc3 import FC3
from models.fc5 import FC5
from torch.utils.data import DataLoader, Subset
import torch
import os
import random
def create_labelwise_dataloaders(dataset, batch_size=32, num_workers=0):
    """
    """
    label_to_indices = {}

  
    for idx, (_, label) in enumerate(dataset):
        label = label.item() if isinstance(label, torch.Tensor) else label  # 确保标签是整数
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)


    labelwise_dataloaders = {}
    for label, indices in label_to_indices.items():

        label_i_data = indices


        other_data = []
        for other_label, other_indices in label_to_indices.items():
            if other_label != label:
                other_data.extend(other_indices)


        num_label_i = len(label_i_data)
        sampled_other_data = random.sample(other_data, num_label_i)


        combined_indices = label_i_data + sampled_other_data
        combined_data = [dataset[idx] for idx in combined_indices]
         

        class_i_data = [(data[0], 1) for data in combined_data[:num_label_i]]
        other_data = [(data[0], 0) for data in combined_data[num_label_i:]]
        
        combined_dataset = class_i_data + other_data
        # class_i_data = [dataset[idx] for idx in label_i_data]
        # class_i_dataset = [(data[0], 1) for data in class_i_data]
        random.shuffle(combined_dataset)
        # 创建 DataLoader
        labelwise_dataloaders[label] = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # labelwise_dataloaders[label] = DataLoader(class_i_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return labelwise_dataloaders
def load_model(model_name, is_modularized=False):
    if model_name == 'fc3':
        model = FC3(is_modularized)
    elif model_name == 'fc5':
        model = FC5(is_modularized)
    else:
        raise ValueError()
    return model

def check_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def load_modules(save_dir,num_modules,model_name,is_modularized = True):
    modules = []
    for i in range(num_modules):
        module = load_model(model_name,is_modularized)
        module.load_state_dict(torch.load(f'{save_dir}/modules_{i}.pth'))
        modules.append(module)

    return modules
def compare_models(model1,model2):
    for (name1,param1), (name2,param2) in zip(model1.named_parameters(),model2.named_parameters()):
        if name1 !=name2 or not torch.equal(param1,param2):
            return False
    return True

    