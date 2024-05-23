import torch
import torch.nn as nn
from torch.nn import functional as F

# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows

# Multi scale fusions
def multi_scale_fusion(stage_outputs):
    num_stages = len(stage_outputs)
    last_stage_index = num_stages - 1
    last_stage_shape = {key: stage_outputs[last_stage_index][key][0].shape for key in stage_outputs[last_stage_index] if key != 'shared'}

    # Step 1: Aggregate shared with task-specific features within each stage
    for stage in stage_outputs:
        shared_feature = stage['shared'][0]
        for task in stage:
            if task == 'shared':
                continue
            task_feature = stage[task][0]
            if task_feature.shape[-2:] != shared_feature.shape[-2:]:
                shared_resized = F.interpolate(shared_feature.unsqueeze(1), size=task_feature.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            else:
                shared_resized = shared_feature
            stage[task] = torch.cat((task_feature, shared_resized), dim=1)

    # Step 2: Aggregate features for each task across all stages
    fused_outputs = {}
    for task in stage_outputs[0]:
        if task == 'shared':
            continue
        task_outputs = [stage[task] for stage in stage_outputs]
        
        # Interpolate each task output to the shape of the last stage task output
        target_shape = last_stage_shape[task][-2:]
        task_outputs = [F.interpolate(output.unsqueeze(1), size=target_shape, mode='bilinear', align_corners=False).squeeze(1) 
                        if output.shape[-2:] != target_shape else output for output in task_outputs]
        
        # Combine the task-specific features across stages
        combined_task_output = torch.stack(task_outputs).mean(dim=0)
        
        fused_outputs[task] = combined_task_output

    return fused_outputs


class MultiScaleAttentionFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiScaleAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, inputs):
        # Concatenate inputs along the temporal dimension for attention
        inputs = torch.stack(inputs)  # Shape: (num_stages, batch_size, input_dim, height, width)
        batch_size,height, width = inputs.shape[1:]
        inputs = inputs.permute(1, 3, 4, 0, 2).reshape(batch_size * height * width, -1, )  # (B*H*W, num_stages, C)
        
        # Apply multi-head attention
        attn_output, _ = self.attention(inputs, inputs, inputs)
        
        # Reshape and combine outputs
        attn_output = attn_output.view(batch_size, height, width, -1, num_channels).permute(0, 3, 4, 1, 2)
        combined_output = attn_output.mean(dim=1)  # Average over stages
        
        return combined_output

def attention_based_fusion(stage_outputs):
    # Instantiate the attention module
    input_dim = stage_outputs[0]['shared'].shape[1]
    output_dim = stage_outputs[-1]['shared'].shape[1]
    fusion_module = MultiScaleAttentionFusion(input_dim, output_dim)
    
    # Aggregate shared and task-specific features within each stage
    for stage in stage_outputs:
        shared_feature = stage['shared']
        for task in stage:
            if task == 'shared':
                continue
            task_feature = stage[task]
            if task_feature.shape[-2:] != shared_feature.shape[-2:]:
                shared_resized = F.interpolate(shared_feature.unsqueeze(1), size=task_feature.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
            else:
                shared_resized = shared_feature
            stage[task] = torch.cat((task_feature, shared_resized), dim=1)
    
    # Apply attention-based fusion for each task
    fused_outputs = {}
    for task in stage_outputs[0]:
        if task == 'shared':
            continue
        task_outputs = [stage[task] for stage in stage_outputs]
        
        # Interpolate each task output to the shape of the last stage task output
        target_shape = stage_outputs[-1][task].shape[-2:]
        task_outputs = [F.interpolate(output.unsqueeze(1), size=target_shape, mode='bilinear', align_corners=False).squeeze(1) 
                        if output.shape[-2:] != target_shape else output for output in task_outputs]
        
        # Apply attention fusion
        combined_task_output = fusion_module(task_outputs)
        
        fused_outputs[task] = combined_task_output

    return fused_outputs

def multi_scale_fusion_selector(stage_outputs, fusion_type='simple'):
    if fusion_type == 'simple':
        return multi_scale_fusion(stage_outputs)
    elif fusion_type == 'attention':
        return attention_based_fusion(stage_outputs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

