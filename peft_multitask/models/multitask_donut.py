
import torch
import torch.nn as nn

from typing import Literal, Optional, Union , Dict
from transformers.modeling_outputs import  Seq2SeqLMOutput 
from copy import deepcopy


# utils functions
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# model class
class MultitaskModel(nn.Module):
    
    def __init__(self, model, decoder_names):
        super(MultitaskModel, self).__init__()
        self.encoder = model.encoder
        self.tasks = decoder_names
        self.decoders_dict = nn.ModuleDict()
        self.loss_dict ={task: 1_000_000 for task in decoder_names}
        
        
        # Initialize decoders from pretrained checkpoints and link them to the shared encoder
        for decoder_name in decoder_names:
            decoder_model = deepcopy(model.decoder)
            decoder_model.encoder = self.encoder
            self.decoders_dict[decoder_name] = decoder_model

    def forward(self, pixel_values , labels ,decoder_input_ids = None ,return_dict: Optional[bool] = None):
        
        assert isinstance(labels , list) , 'At least two labels are required for multitask training'
        
        labels_dict = {task : labels[i] for i,task in enumerate(self.tasks)}
        
        # Pass input data through the shared encoder
        encoder_outputs = self.encoder(pixel_values)
        
        tasks_outputs = dict()
        
        for task in self.tasks :
            
            labels = labels_dict[task]
            
            # Get the appropriate encoder task specific output 
            encoder_output = encoder_outputs[task]

            # Get the last hidden state for the output
            encoder_hidden_states = encoder_output.last_hidden_state

            # Get the appropriate decoder for the task
            decoder_model = self.decoders_dict[task]

            # Create the decoder input
            if (labels is not None) and (decoder_input_ids is None):
                decoder_input_ids = shift_tokens_right(
                    labels, decoder_model.config.pad_token_id, decoder_model.config.decoder_start_token_id
                    )


            # Pass encoder output through the decoder
            decoder_outputs = decoder_model( input_ids  = decoder_input_ids,
                                            encoder_hidden_states = encoder_hidden_states)

            # Compute the loss
            loss = None
            if labels is not None:
                logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, decoder_model.config.vocab_size), labels.reshape(-1))

            self.loss_dict[task] = loss
            
            task_output = Seq2SeqLMOutput(loss=loss,
                               logits=decoder_outputs.logits,
                               past_key_values=decoder_outputs.past_key_values,
                               decoder_hidden_states=decoder_outputs.hidden_states,
                               decoder_attentions=decoder_outputs.attentions,
                               cross_attentions=decoder_outputs.cross_attentions,
                               encoder_last_hidden_state= encoder_output.last_hidden_state,
                               encoder_hidden_states = encoder_output.hidden_states,
                               encoder_attentions=encoder_output.attentions)
            
            tasks_outputs.update({task : task_output})
        
        return tasks_outputs
    
    
    @torch.no_grad()
    def generate(self, pixel_values , max_length = 512 , device = 'cuda'):
        
        # Pass input images through the shared encoder
        encoder_outputs = self.encoder(pixel_values)
        
        # get the decoder output
        outputs = dict()
        
        
        for task in self.tasks :
            
            # Get the appropriate encoder task specific output 
            encoder_output = encoder_outputs[task]
            
            # Get the last hidden state for the output
            encoder_hidden_states = encoder_output.last_hidden_state
            
            # Get the appropriate decoder for the task
            decoder_model = self.decoders_dict[task]
            
            #inpts for the decoder
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            
            # Initialize an empty list to store the generated tokens
            generated_tokens = []

            # Generate text token by token until reaching the maximum length or generating the end-of-sequence token
            for _ in range(max_length):
                        # Generate the next token
                        decoder_outputs = decoder_model(input_ids=decoder_input_ids, encoder_hidden_states=encoder_hidden_states)
                        next_token_logits = decoder_outputs.logits[:, -1, :]
                        next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(-1)


                        # Append the generated token to the list
                        generated_tokens.append(next_token_id.item())

                        # Check if the end-of-sequence token is generated
                        if next_token_id.item() == decoder_model.config.eos_token_id:
                            break

                        # Update the decoder input for the next iteration
                        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
                        
            # Convert the generated token IDs to text using the model's tokenizer
            sequence = processor.batch_decode(decoder_input_ids)[0]
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
            outputs.update({task:sequence})
            
        return outputs

