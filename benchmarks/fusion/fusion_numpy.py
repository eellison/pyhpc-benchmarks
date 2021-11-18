"""
==========================================================================
  in-situ density, dynamic enthalpy and derivatives
  from Absolute Salinity and Conservative
  Temperature, using the computationally-efficient 48-term expression for
  density in terms of SA, CT and p (IOC et al., 2010).
==========================================================================
"""

import np

def batch_norm_add_relu(input, weight, bias, running_mean, running_var, following_add):
    momentum = .1
    eps = .5
    inv_var = 1 / np.sqrt(running_var + eps)
    alpha = inv_var * weight
    beta = bias - running_mean * alpha
    output = input * np.expand_dims(np.expand_dims(alpha + beta, axis=-1), axis=-1)
    output_add = output + following_add
    return jax.nn.relu(output_add)

def run(inputs, device='cpu'):
    return batch_norm_add_relu(*inputs)