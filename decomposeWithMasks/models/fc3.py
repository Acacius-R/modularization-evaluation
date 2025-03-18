import torch
import torch.nn as nn

from torch.utils.data import DataLoader

class FC3(nn.Module):
    def __init__(self,is_modularized=False):
        super(FC3, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 49)  
        self.fc2 = nn.Linear(49, 10)  
        self.softmax = nn.Softmax(dim=1)
        self.is_modularized = is_modularized
        if is_modularized:
            self.head = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Linear(10, 2),
            nn.Softmax(dim=1)
        )
            # self.modularized_params = self.fc4.parameters()
            

    def forward(self, x):
        x = self.flatten(x)  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        
        if self.is_modularized:
            x = self.head(x)
        else:
            x = self.softmax(x)  
        return x

    
    
