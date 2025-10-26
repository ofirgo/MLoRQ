"""
 Some functions in this file is copied from https://github.com/yhhhli/BRECQ
 and modified for this project's needs.
"""
import torch

from compression.quantization.helpers import ste_floor, ste_clip


def get_soft_targets(alpha, gamma, zeta):
    return torch.clamp(torch.sigmoid(alpha) * (zeta - gamma) + gamma, 0, 1)


def modified_soft_quantization(w, alpha, n_bits, delta, gamma, zeta, zero_point, training=False, eps=1e-8,
                               gradient_factor=1.0):
    w = w.detach()
    w_floor = ste_floor(w / (delta + eps), gradient_factor=gradient_factor)

    if training:
        w_int = w_floor + get_soft_targets(alpha, gamma, zeta)
    else:
        w_int = w_floor + (alpha >= 0).float()

    w_quant = ste_clip(w_int + zero_point, 0, 2**n_bits - 1)
    w_float_q = (w_quant - zero_point) * delta

    return w_float_q
