import torch

import torch
import sys
'''
class PrefixBlock(torch.nn.Module):
  r
  A block to encode prefix

  Input shape: (batch-size, prefix-length)

  Output shape: (batch-size, prefix-length, 2 * hidden)
  
  def __init__(self, config):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.embedding = torch.nn.Embedding(config.pre_seq_len, 2 * config.hidden_size)
    self.gate = torch.nn.Sequential(
        torch.nn.Linear(2 * config.hidden_size, 1, bias=True),
        torch.nn.Sigmoid()
    )

  def forward(self, prefix: torch.tensor):
    
    prefix_tokens = self.embedding(prefix)
    
    key, value = torch.split(prefix_tokens, self.hidden_size, dim=-1)
    single_mask = self.gate(prefix_tokens)
    # past_key_value = self.linear(prefix_tokens)
    key_mask = torch.cat([torch.log(single_mask) for _ in range(self.hidden_size)], dim=-1)
    key = key + key_mask
    return torch.cat([key, value], dim=-1)
'''


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        #self.prefix_ada = config.prefix_ada
        self.num_hidden_layers = config.num_hidden_layers
        
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values