"""
 Some functions in this file is copied from https://github.com/zkkli/RepQ-ViT or https://github.com/zysxmu/ERQ
 and modified for this project's needs.
"""
import numpy as np
import torch.linalg


def ste_round(x: torch.Tensor, gradient_factor=1.0) -> torch.Tensor:
    """
    Return the rounded values of a tensor.
    """
    return (torch.round(x) - x * gradient_factor).detach() + x * gradient_factor


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def uniform_quantize_min_max(x, max, min, n_bits):
    delta = (max - min) / (2 ** n_bits - 1)
    zero_point = (- min / delta).round()
    # we assume weight quantization is always signed
    x_int = torch.round(x / delta)
    x_quant = torch.clamp(x_int + zero_point, 0, 2 ** n_bits - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q

def uniform_quantization(x, delta, zero_point, n_levels):
    x_int = torch.round(x / delta) + zero_point
    x_quant = torch.clamp(x_int, 0, n_levels - 1)
    x_dequant = (x_quant - zero_point) * delta
    return x_dequant

def search_weights_scale_perc(x: torch.Tensor, n_bits, hessian=None, channel_wise: bool = False, x_ref=None,
                              x_complement=None,
                              new_mode=False):
    if new_mode:
        x_clone = x.clone().detach()
        if channel_wise:
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) in [2, 4]:
                pass
            elif len(x.shape) == 3:
                x_clone = x_clone.permute(2, 0, 1)
            else:
                raise NotImplementedError

            if hessian is not None:
                if len(x.shape) == 3:
                    c_hessian = hessian.clone().detach().permute(2, 0, 1)
                else:
                    c_hessian = hessian.clone().detach()
                c_hessian = c_hessian.reshape(n_channels, -1)
            else:
                c_hessian = None

            x_clone = x_clone.reshape(n_channels, -1)
        else:
            x_clone = x_clone.reshape(1, -1)

        x_max = x_clone.abs().max(dim=1)[0]

        delta = x_max.clone()
        zero_point = x_max.clone()

        best_score = x_max.clone().fill_(1e10)
        pct_dict = {8: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    7: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    6: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    5: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    4: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    3: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    2: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1]}

        for pct in pct_dict.get(n_bits):
            new_max = torch.quantile(x_clone, pct, dim=1, keepdim=True)
            new_min = torch.quantile(x_clone, 1.0 - pct, dim=1, keepdim=True)

            x_q = uniform_quantize_min_max(x_clone, new_max, new_min, n_bits)

            if x_ref is None:
                if c_hessian is None:
                    score = (x_clone - x_q).abs().pow(2).mean(1)
                else:
                    score = (torch.sqrt(c_hessian) * ((x_clone - x_q).abs())).pow(2).mean(1)
            else:
                if c_hessian is None:
                    score = (x_ref - x_q @ x_complement).abs().pow(2).mean(1)
                else:
                    score = (torch.sqrt(c_hessian) * ((x_ref - x_q @ x_complement).abs())).pow(2).mean(1)

            better_inds = score < best_score
            best_score[better_inds] = score[better_inds]
            delta[better_inds] = (new_max - new_min)[better_inds, 0] / (2 ** n_bits - 1)
            zero_point[better_inds] = (- new_min[:, 0] / delta)[better_inds].round()

        if len(x.shape) == 4:
            delta = delta.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:
            delta = delta.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        elif len(x.shape) == 3:
            delta = delta.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        else:
            raise NotImplementedError

        return delta, zero_point
    else:
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if hessian is not None:
                    if len(x.shape) == 3:
                        c_hessian = hessian[:, :, c]
                    else:
                        c_hessian = hessian[c]
                else:
                    c_hessian = None

                x_ref_c = None
                if len(x.shape) == 3:
                    if x_ref is not None:
                        x_ref_c = x_ref[:, :, c]
                    delta[c], zero_point[c] = search_weights_scale_perc(x_clone[:, :, c], n_bits=n_bits,
                                                                        channel_wise=False, x_ref=x_ref_c,
                                                                        x_complement=x_complement,
                                                                        hessian=c_hessian, new_mode=new_mode)
                else:
                    if x_ref is not None:
                        x_ref_c = x_ref[c]
                    delta[c], zero_point[c] = search_weights_scale_perc(x_clone[c], n_bits=n_bits,
                                                                        channel_wise=False, x_ref=x_ref_c,
                                                                        x_complement=x_complement,
                                                                        hessian=c_hessian, new_mode=new_mode)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
            else:
                raise NotImplementedError
        else:
            x_clone = x.clone().detach()
            if x_ref is not None:
                x_ref = x_ref.clone().detach()
                x_complement = x_complement.clone().detach()

            best_score = 1e+10
            pct_dict = {8: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                        7: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                        6: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                        5: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                        4: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                        3: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                        2: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1]}

            for pct in pct_dict.get(n_bits):
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                x_q = uniform_quantize_min_max(x_clone, new_max, new_min, n_bits)

                if x_ref is None:
                    if hessian is None:
                        score = lp_loss(x_clone, x_q, p=2, reduction='all')
                    else:
                        score = (torch.sqrt(hessian) * ((x_clone - x_q).abs())).pow(2).mean()
                else:
                    if hessian is None:
                        score = lp_loss(x_ref, x_q @ x_complement, p=2, reduction='all')
                    else:
                        score = (torch.sqrt(hessian) * ((x_ref - x_q @ x_complement).abs())).pow(2).mean()
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** n_bits - 1)
                    zero_point = (- new_min / delta).round()
        return delta, zero_point
