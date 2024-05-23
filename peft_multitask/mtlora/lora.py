import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class TA_LoRA(nn.Module):
    def __init__(self, linear_layer, rank, lora_alpha=1.0, lora_dropout=0.0):
        super(TA_LoRA, self).__init__()
        self.linear_layer = linear_layer
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        return self.linear_layer(x) + self.lora_dropout(x @ self.lora_A @ self.lora_B) * self.lora_alpha
    
    
    
class TS_LoRA(nn.Module):
    def __init__(self, linear_layer, rank, lora_alpha=1.0, lora_dropout=0.0, tasks=['task1', 'task2']):
        super(TS_LoRA, self).__init__()
        self.linear_layer = TA_LoRA(linear_layer, rank, lora_alpha, lora_dropout)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.tasks = tasks
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.lora_A = nn.ParameterDict({task: nn.Parameter(torch.zeros(in_features, rank)) for task in tasks})
        self.lora_B = nn.ParameterDict({task: nn.Parameter(torch.zeros(rank, out_features)) for task in tasks})

        for task in tasks:
            nn.init.kaiming_uniform_(self.lora_A[task], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[task])

    def forward(self, x, task=True):
        shared_output = self.linear_layer(x)
        task_outputs = {}
        
        if task is not None:
            for t in self.tasks:
                task_output = shared_output + self.lora_dropout(x @ self.lora_A[t] @ self.lora_B[t]) * self.lora_alpha
                task_outputs[t] = task_output
        
        return {'shared': shared_output, **task_outputs}



def replace_linear_with_lora(module, ta_lora_config, ts_lora_config, is_last_layer=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if is_last_layer:
                ts_lora_layer = TS_LoRA(child, **ts_lora_config)
                setattr(module, name, ts_lora_layer)
            else:
                new_layer = TA_LoRA(child, **ta_lora_config)
                setattr(module, name, new_layer)
                
        elif name != 'output' :
            replace_linear_with_lora(child, ta_lora_config, ts_lora_config, is_last_layer= False)
        elif name == 'output' :
            replace_linear_with_lora(child, ta_lora_config, ts_lora_config, is_last_layer)
            
    return module


def get_mtlora_encoder(encoder, ta_lora_config, ts_lora_config):
    for stage in encoder.layers:
        for block in stage.blocks[:-1]:
            replace_linear_with_lora(block, ta_lora_config, ts_lora_config)
        # Ensure the last block of each stage uses TS_LoRA
        replace_linear_with_lora(stage.blocks[-1], ta_lora_config, ts_lora_config, is_last_layer=True)
    return encoder
