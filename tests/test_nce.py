import torch
from torch import nn
from mup import MuReadout, MuSharedReadout


class LMHead(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()
        use_mup = config.get("use_mup", False)
        if embedding is not None:
            if use_mup:
                self.linear = MuSharedReadout(embedding.weight, bias=False)
            else:
                self.linear = nn.Linear(
                    config["d_model"], config["vocab_size"]
                )
                self.linear.weight = embedding.weight
        else:
            if use_mup:
                self.linear = MuReadout(
                    config["d_model"], config["vocab_size"]
                )
            else:
                self.linear = nn.Linear(
                    config["d_model"], config["vocab_size"]
                )

    def forward(self, x):
        return self.linear(x)
    
    def sparse_forward(self, x, idx):
        # TODO check if there are sparse operations
        # to avoid copying the weights
        weight = self.linear.weight[idx]
        bias = self.linear.bias[idx] if self.linear.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)
    
    def nce(self, x, targets):
        """
        x: (batch_size, seq_len, d_model)
        targets: (batch_size, seq_len)
        """
        batch_size = x.shape[0]
        # Sample negative indices
        
        # compute logits
        
        
        
        

"""
LMhead forward can now compute logits for small set of negative samples instead of the entire vocabulary.
1. Sample negative indices
    - Here we shall use all the targets we have to compute anyway 
    targets = (batch_size, seq_len)

2. Compute logits for negative indices
3. Compute logits for positive index
4. Compute loss
"""