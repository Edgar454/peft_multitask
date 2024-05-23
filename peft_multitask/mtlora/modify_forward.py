
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from transformers.utils import ModelOutput

from .forward_utils import window_partition , window_reverse , multi_scale_fusion_selector


# Defining the output data classes 

@dataclass
# Copied from transformers.models.swin.modeling_swin.SwinEncoderOutput with Swin->DonutSwin
class DonutSwinEncoderOutput(ModelOutput):  
    """
    DonutSwin encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

        
@dataclass
# Copied from transformers.models.swin.modeling_swin.SwinModelOutput with Swin->DonutSwin
class DonutSwinModelOutput(ModelOutput):
    """
    DonutSwin model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


#===================== Modifying the output layer forward =====================================================================

def output_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        if isinstance(hidden_states , torch.Tensor):
            hidden_states = self.dropout(hidden_states)
        elif isinstance(hidden_states, nn.ModuleDict) :
            hidden_states = {task : self.dropout(hidden_states[task] for task in hidden_states )}
        return hidden_states


#===================== Modifying the block forward =====================================================================
def modified_forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Dict[str, Any]:
        
    if not always_partition:
        self.set_shift_and_window_size(input_dimensions)
    else:
        pass
        
    height, width = input_dimensions
    batch_size, _, channels = hidden_states.size()
    shortcut = hidden_states

    hidden_states = self.layernorm_before(hidden_states)
    hidden_states = hidden_states.view(batch_size, height, width, channels)

    # pad hidden_states to multiples of window size
    hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
    _, height_pad, width_pad, _ = hidden_states.shape

    # cyclic shift
    if self.shift_size > 0:
        shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_hidden_states = hidden_states

    # partition windows
    hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
    hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
    attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
    
    if attn_mask is not None:
        attn_mask = attn_mask.to(hidden_states_windows.device)

    attention_outputs = self.attention(
        hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
    )
    
    attention_output = attention_outputs[0]
    attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
    shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

    # reverse cyclic shift
    if self.shift_size > 0:
        attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        attention_windows = shifted_windows

    was_padded = pad_values[3] > 0 or pad_values[5] > 0
    if was_padded:
        attention_windows = attention_windows[:, :height, :width, :].contiguous()

    attention_windows = attention_windows.view(batch_size, height * width, channels)
    hidden_states = shortcut + self.drop_path(attention_windows)

    layer_output = self.layernorm_after(hidden_states)
    layer_output = self.intermediate(layer_output)
    lora_output = self.output(layer_output)  # ts_lora_output is a dict
    
        
    if isinstance(lora_output , torch.Tensor):
        layer_output = hidden_states + self.output(layer_output)
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        
    elif isinstance(lora_output, Dict) :
        final_output = {task: hidden_states + lora_output[task] for task in lora_output}
        
        # ======================à reviser hypothétiquement============================#
        final_output = {task: self.layernorm_after(final_output[task]) for task in final_output}

        if output_attentions:
            layer_outputs = {task: (final_output[task], attention_outputs[1]) for task in final_output}
        else:
            layer_outputs = final_output
    
    return layer_outputs

#===================== Modifying the stage forward =====================================================================

def modified_stage_forward(
    self,
    hidden_states: torch.Tensor,
    input_dimensions: Tuple[int, int],
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    always_partition: Optional[bool] = False,
) -> Dict[str, Any]:
    
    height, width = input_dimensions
    for i, layer_module in enumerate(self.blocks):
        layer_head_mask = head_mask[i] if head_mask is not None else None

        layer_outputs = layer_module(
            hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
        )

        if isinstance(layer_outputs, dict):
            hidden_states = {task: layer_outputs[task] for task in layer_outputs}
        else:
            hidden_states = layer_outputs[0]

    hidden_states_before_downsampling = hidden_states
    
    

    if self.downsample is not None:
        height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
        output_dimensions = (height, width, height_downsampled, width_downsampled)
        
        if isinstance(hidden_states_before_downsampling, dict):
            hidden_states = {task: self.downsample(hidden_states_before_downsampling[task], input_dimensions) for task in hidden_states_before_downsampling}
        else:
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
    else:
        output_dimensions = (height, width, height, width)

    if isinstance(hidden_states, dict):
        stage_outputs = {task: (hidden_states[task], hidden_states_before_downsampling[task], output_dimensions) for task in hidden_states}
        if output_attentions:
            for task in stage_outputs:
                stage_outputs[task] += layer_outputs[task][1:]
        
    else:
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
        if output_attentions:
            stage_outputs += layer_outputs[1:]
    return stage_outputs


#===================== Modifying the encoder forward =====================================================================

def modified_encoder_forward(
    self,
    hidden_states: torch.Tensor,
    input_dimensions: Tuple[int, int],
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    output_hidden_states_before_downsampling: Optional[bool] = False,
    always_partition: Optional[bool] = False,
    return_dict: Optional[bool] = True,
) -> Union[Tuple, DonutSwinEncoderOutput]:
    all_hidden_states = () if output_hidden_states else None
    all_reshaped_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    if output_hidden_states:
        batch_size, _, hidden_size = hidden_states.shape
        # rearrange b (h w) c -> b c h w
        reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
        reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
        all_hidden_states += (hidden_states,)
        all_reshaped_hidden_states += (reshaped_hidden_state,)
        
    stage_outputs = []
    for i, layer_module in enumerate(self.layers):
        layer_head_mask = head_mask[i] if head_mask is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.__call__,
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
                always_partition,
            )
        else:
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

        if isinstance(layer_outputs, dict):
            hidden_states = layer_outputs['shared'][0] 
            hidden_states_before_downsampling = layer_outputs['shared'][1] 
            output_dimensions = layer_outputs['shared'][2] 
            input_dimensions =  (output_dimensions[-2], output_dimensions[-1]) 
        else:
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            
        stage_outputs.append(layer_outputs)

        if output_hidden_states:
            if isinstance(hidden_states, dict):
                for task in hidden_states:
                    batch_size, _, hidden_size = hidden_states[task].shape
                    reshaped_hidden_state = hidden_states[task].view(batch_size, *input_dimensions[task], hidden_size)
                    reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                    all_hidden_states += (hidden_states[task],)
                    all_reshaped_hidden_states += (reshaped_hidden_state,)
            else:
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

        if output_attentions:
            if isinstance(layer_outputs, dict):
                for task in layer_outputs:
                    all_self_attentions += layer_outputs[task][3:]
            else:
                all_self_attentions += layer_outputs[3:]
                
    fused_outputs = multi_scale_fusion_selector(stage_outputs, fusion_type='simple')
    
    if not return_dict:
        if isinstance(fused_outputs, dict):
            return {task: tuple(v for v in [hidden_states[task], all_hidden_states, all_self_attentions] if v is not None) for task in hidden_states}
        else:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

    if isinstance(fused_outputs, dict):
        return {task: DonutSwinEncoderOutput(
                    last_hidden_state=fused_outputs[task],
                    hidden_states=all_hidden_states,
                    attentions=all_self_attentions,
                    reshaped_hidden_states=all_reshaped_hidden_states,
                ) for task in fused_outputs}
    else:
        return DonutSwinEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )

#===================== Modifying the swin forward =====================================================================

def modified_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    bool_masked_pos: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, DonutSwinModelOutput]:
    r"""
    bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, len(self.config.depths))

    embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

    encoder_outputs = self.encoder(
        embedding_output,
        input_dimensions,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if isinstance(encoder_outputs, dict):
        sequence_output = {task: encoder_outputs[task][0] for task in encoder_outputs}

        pooled_output = {task: None for task in encoder_outputs}
        
        if self.pooler is not None:
            for task in encoder_outputs:
                seq_out_task = sequence_output[task]
                # Ensure seq_out_task has 3 dimensions (batch_size, sequence_length, hidden_size)
                if seq_out_task.dim() == 3:
                    seq_out_task = seq_out_task.transpose(1, 2)  # (batch_size, hidden_size, sequence_length)
                elif seq_out_task.dim() == 2:
                    seq_out_task = seq_out_task.unsqueeze(0).transpose(1, 2)  # (1, hidden_size, sequence_length)
                pooled_output[task] = self.pooler(seq_out_task)
                pooled_output[task] = torch.flatten(pooled_output[task], 1)

        if not return_dict:
            output = {task: (sequence_output[task], pooled_output[task]) + encoder_outputs[task][1:] for task in encoder_outputs}
            return output
        
        
        return {task: DonutSwinModelOutput(
            last_hidden_state=sequence_output[task],
            pooler_output=pooled_output[task],
            hidden_states=encoder_outputs[task].get('hidden_states'),
            attentions=encoder_outputs[task].get('attentions'),
            reshaped_hidden_states=encoder_outputs[task].get('reshaped_hidden_states'),
        ) for task in encoder_outputs}

    else:
        sequence_output = encoder_outputs[0]

        pooled_output = None
        if self.pooler is not None:
            sequence_output = sequence_output.transpose(1, 2)  # (batch_size, hidden_size, sequence_length)
            pooled_output = self.pooler(sequence_output)
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        return DonutSwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )


# functions to change the specifics forward functions
import types

def replace_output_forward_method_with_modified(output):
    output.forward = types.MethodType(output_forward, output)

def replace_forward_method_with_modified(block):
    block.forward = types.MethodType(modified_forward, block)
  
def replace_stage_forward_method_with_modified(stage):
    stage.forward = types.MethodType(modified_stage_forward, stage)
    
def replace_encoder_forward_method_with_modified(encoder):
    encoder.forward = types.MethodType(modified_encoder_forward, encoder)
    
def replace_donut_encoder_forward_method(swin):
    encoder.forward = types.MethodType(modified_forward, swin)
    
    
# function to wrap it all togetget

def replace_forward(swin_encoder):

    replace_donut_encoder_forward_method(swin_encoder)
    
    donut_encoder = swin_encoder.encoder
    replace_encoder_forward_method_with_modified(donut_encoder)
    
    for stage in donut_encoder.layers:
        replace_stage_forward_method_with_modified(stage)
        for block in stage.blocks:
            replace_forward_method_with_modified(block)
            replace_output_forward_method_with_modified(block.output)

