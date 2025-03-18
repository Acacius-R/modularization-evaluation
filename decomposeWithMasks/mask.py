import torch
import torch.nn as nn
import torch.nn.init as init
from utils import *
class MaskModel(nn.Module):
    def __init__(self, model,threshold,model_name):
        super(MaskModel,self).__init__()
        self.origin_model = model
        self.masked_model = load_model(model_name,True)
        self.scores={}
        self.percent = threshold

        for name, param in self.origin_model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # initialize with Kaiming Normal
                score = torch.zeros_like(param)
                score.requires_grad = True
                init.kaiming_normal_(score, mode='fan_in', nonlinearity='relu')
                self.scores[name] = nn.Parameter(score)
                self.masked_model.state_dict()[name].data = param.data

    def apply_mask(self):
        for name,param in self.masked_model.named_parameters():
            if name in self.scores:
                score = self.scores[name]
                
                k = int(self.percent * score.numel())
                if k == 0:
                    mask = torch.zeros_like(score).bool()
                else:
                    
                    threshold = torch.topk(score.flatten(), k)[0][-1]
                    mask = (score >= threshold)
               
                param.data = self.origin_model.state_dict()[name].data.to(param.device) * mask.float().to(param.device)

            

    def forward(self,x):
       
        return self.masked_model(x)
    
    def save_modules(self,save_dir):
        torch.save(self.masked_model.state_dict(),save_dir)

    