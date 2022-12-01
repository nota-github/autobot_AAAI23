import torch
import torch.nn as nn
import numpy as np
from utils.calc_flops import get_flops

class Bottleneck(nn.Module):
    """
    The Attribution Bottleneck.
    Is inserted in a existing model to suppress information, parametrized by a suppression mask alpha.
    
    Let's M be the module before the bottleneck.
    Args:
        - module_type: type of M (Conv2d, Linear, BatchNorm, etc...)
        - info: info on M necessary to compute the FLOPS
        - lambda_in: list of lambda masks that restrict the input flow of M
        - lamb_out: list of of lambda masks that restrict the output flow of M

    Remarks:
        - if you don't want the module output to be pruned, use lamb_out=None
        - if you don't want the module input to be pruned, use lamb_in=None
        - if module_type is BatchNorm or ReLu, then lambda_in == lamb_out
        - if there is no skip connections, then len(lambda_in) == len(lamb_out) = 1,
        meaning that no masks are concatenated.
    """
    def __init__(self, module_type, info, lambda_in_idx, lambda_out_idx, alphas, forward_with_mask=True):
        super().__init__()
        self.module_type = module_type
        self.info = info
        self.lambda_in_idx = lambda_in_idx
        self.lambda_out_idx = lambda_out_idx
        self.alphas = alphas
        self.update_lambdas()
        self.forward_with_mask = False if lambda_out_idx is None else forward_with_mask

    def update_lambdas(self):
        self.lambda_in = None if self.lambda_in_idx is None else [self.alphas.get_lambda(i) for i in self.lambda_in_idx]
        self.lambda_out = None if self.lambda_out_idx is None else [self.alphas.get_lambda(i) for i in self.lambda_out_idx]

    def forward(self, r):
        """ Restrict the information from r by reducing the signal amplitude, using the mask alpha """
        if not self.forward_with_mask: return r
        prev = 0
        for i, lmb in enumerate(self.lambda_out):
            partial_r = r[:, prev:prev+lmb.size(0)]#.to(lmb.device)
            lamb = lmb.unsqueeze(1).unsqueeze(1)
            partial_out = lamb*partial_r
            z = torch.cat((z, partial_out), dim=1) if i>0 else partial_out
            prev += lmb.size(0)
        return z

    def compute_flops(self, ignore_mask=False):
        if ignore_mask:
            return get_flops(self.module_type, self.info, None, None)
        return get_flops(self.module_type, self.info, self.lambda_in, self.lambda_out)


class UniqueAlphaMasks(nn.Module):
    """
    If you input a new init_lamb, it creates a new alpha mask
    parameter (nn.Parameter) from it, and it return the associated lambda.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.alphas = []
        self.sigmoid = nn.Sigmoid()

    def len(self):
        return len(self.alphas)

    def get_params(self, used_masks=None):
        """
        Return the list of params (for training)
        """
        return self.alphas if used_masks is None else [self.alphas[i] for i in range(self.len()) if used_masks[i]]

    def get_lambda(self, pos):
        return self.sigmoid(self.alphas[pos])

    def set_lambda(self, init_lamb, pos):
        if isinstance(init_lamb, list):
            init_lamb = torch.tensor(init_lamb, dtype=torch.float32)
        
        init_alpha = 1*(-torch.log((1 / (init_lamb + 1e-8)) - 1)).to(self.device)
        self.alphas[pos].requires_grad = False
        self.alphas[pos].copy_(init_alpha.detach().clone())

    def insert(self, pos, init_lamb):
        init_alpha = (-torch.log((1 / (init_lamb + 1e-8)) - 1)).to(self.device)
        alpha = nn.Parameter(init_alpha.detach().clone())
        self.alphas.insert(pos, alpha)

    def append(self, init_lamb):
        init_alpha = (-torch.log((1 / (init_lamb + 1e-8)) - 1)).to(self.device)
        alpha = nn.Parameter(init_alpha.detach().clone())
        self.alphas.append(alpha)

