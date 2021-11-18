"""
==========================================================================
  in-situ density, dynamic enthalpy and derivatives
  from Absolute Salinity and Conservative
  Temperature, using the computationally-efficient 48-term expression for
  density in terms of SA, CT and p (IOC et al., 2010).
==========================================================================
"""

import numpy as np
import torch


@torch.jit.script
def batch_norm_add_relu(input, weight, bias, running_mean, running_var, following_add):
    momentum = .1
    eps = .5
    inv_var = torch.rsqrt(running_var + eps)
    alpha = inv_var * weight
    beta = bias - running_mean * alpha
    output = input * (alpha + beta).unsqueeze(1).unsqueeze(1)
    output_add = output + following_add
    return torch.relu(output_add)


def prepare_inputs(inputs, device):
    torch._C_jit_texpr_reductions_enabled(False)
    out = [torch.as_tensor(a, device='cuda' if device == 'gpu' else 'cpu') for a in inputs]
    if device == 'gpu':
        torch.cuda.synchronize()
    return out


def run(inputs, device='cpu'):
    with torch.no_grad():
        out = batch_norm_add_relu(*inputs)
    if device == 'gpu':
        torch.cuda.synchronize()
    return out
