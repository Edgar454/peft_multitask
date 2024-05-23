
from .modify_forward import replace_forward
from .lora import get_mtlora_encoder

from torch import nn

class Mtlora_encoder(nn.Module):
    
    def __init__(self ,
                 swin_endoder : nn.Module ,
                 ta_lora_config,
                 ts_lora_config):
        
        super(Mtlora_encoder , self).__init__()
        
        self.swin_encoder = swin_endoder
        get_mtlora_encoder(self.swin_encoder.encoder , ta_lora_config, ts_lora_config)
        replace_forward(self.swin_encoder)
        
        self.get_peft_model(self.swin_encoder)

        
    def get_peft_model(self, model):
        """
        Deactivates all layers in the model except the LoRA layers by setting `requires_grad` to False.
        """
        for name, param in model.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
                
    def load_pretrained(self, load_path):
        """
        Loads the LoRA layers from a specified checkpoint.
        """
        
        model = self.swin_encoder
        lora_layers = torch.load(load_path)
        model_dict = model.state_dict()
        model_dict.update(lora_layers)
        model.load_state_dict(model_dict)

    
    def get_memory_footprint(self ,model):
        """
        Returns the memory footprint of the model in bytes.
        """
        total_memory = sum(param.numel() * param.element_size() for param in model.parameters())
        return total_memory


                
    def print_model_info(self):
        """
        Prints the number of trainable parameters, total parameters, percentage of trainable parameters,
        and memory footprint of the model.
        """
        model = self.swin_encoder
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        percentage_trainable = (trainable_params / total_params) * 100

        memory_footprint = self.get_memory_footprint(model)
        memory_footprint_mb = memory_footprint / (1024 ** 2)  # Convert bytes to megabytes

        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Percentage of Trainable Parameters: {percentage_trainable:.2f}%")
        print(f"Memory Footprint: {memory_footprint_mb:.2f} MB")

