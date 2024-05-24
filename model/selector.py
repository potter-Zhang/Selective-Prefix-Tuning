import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PseudoPrefixEncoder(nn.Module):
    '''
        input: [batch_size, pre_seq_len]
        output: [batch_size, pre_seq_len, hidden_size]
    '''
    def __init__(self, config):
        super().__init__()
        self.key_prefix = nn.Embedding(config.pre_seq_len, config.hidden_size)
        self.value_prefix = nn.Embedding(config.pre_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.key_dropout_prob)
    
    def forward(self, prefix_ids):
        return self.dropout(self.key_prefix(prefix_ids)), self.value_prefix(prefix_ids)
 
class PrefixSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer('tau', torch.tensor(config.tau))
        
        self.fn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.pre_seq_len = config.pre_seq_len
        self.dropout = nn.Dropout(config.scores_dropout_prob)
    
    def forward(self, attention_scores):
        B, H, seq, _ = attention_scores.size()
        prefix_scores = attention_scores[:, :, :, :self.pre_seq_len]
        
        prefix_scores = prefix_scores * self.tau
        mask = self.fn(self.dropout(prefix_scores))
        ones = attention_scores.new_ones(B, H, seq, seq)
        
        attention_scores = attention_scores + torch.log(torch.cat([mask + 1e-30, ones], dim=-1))
        
        attention_probs = self.softmax(attention_scores)
        #print(self.tau)
        return attention_probs
 
def compute_regular_loss(model, params, loss):
    num = model.n_layer * (model.pre_seq_len * (model.pre_seq_len - 1))
    regular_loss = None
    #print("hhhh")
    for name, param in params:
        if "value_prefix" in name or "key_prefix" in name:
            normalize_prefix = param / torch.sqrt(torch.sum(param * param, dim=-1, keepdim=True) + 1e-8)
            layer_regular_loss = (torch.sum(torch.abs(normalize_prefix @ normalize_prefix.t())) - model.pre_seq_len) / 2
            
            if regular_loss:
                regular_loss += layer_regular_loss
            else:
                regular_loss = layer_regular_loss
    regular_loss = regular_loss / num
    loss += model.alpha * regular_loss
    return loss