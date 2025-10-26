"""
 Some functions in this file is copied from https://github.com/zkkli/RepQ-ViT or https://github.com/zysxmu/ERQ
 and modified for this project's needs.
"""

import datetime
import os
import pickle
import shutil
from copy import deepcopy
from math import sqrt
from typing import List, Dict, Type, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, Node
from torch.nn import LayerNorm, Conv2d, Module

from compression.configs.compression_config import CompressionConfig
from compression.module.compression_wrapper import CompressionWrapper
from compression.quantization.quantization import lp_loss
from constants import CALL_FUNCTION, CALL_MODULE, CALL_METHOD, ACTIVATION_QUANT_STR, DEVICE, FLOAT_BIT_WIDTH, \
    MP_BIT_WIDTH, REDUNDANT_OPERATORS

from constants import PRIMARY_DEVICE


def im2col(input_data, kernel_size, stride, padding):
    input_padded = F.pad(input_data, (padding, padding, padding, padding))
    batch_size, channels, height, width = input_padded.shape

    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1

    cols = torch.zeros(batch_size, channels, kernel_size, kernel_size, out_height, out_width, device=input_data.device)

    for y in range(kernel_size):
        y_max = y + stride * out_height
        for x in range(kernel_size):
            x_max = x + stride * out_width
            cols[:, :, y, x, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride]

    cols = cols.permute(0, 4, 5, 1, 2, 3).reshape(batch_size * out_height * out_width, -1)
    return cols


def myhook(module, input, output):
    if module.store_input == None:
        module.store_input = []
        module.store_output = []
    module.store_input.append(input[0].cpu().detach())
    module.store_output.append(output.cpu().detach())


def hook_fp_act(q_model, calib_data, args):
    hooks = []

    skip_rr = False

    for n, m in q_model.named_modules():
        if isinstance(m, CompressionWrapper):
            if 'reduction' in n:
                print(f"Skipping reduction layer in RR: {n}")
            else:
                if skip_rr:
                    print(f"Skipping RR in layer: {n}")
                    skip_rr = False
                else:
                    hooks.append(m.register_forward_hook(myhook))
        elif isinstance(m, LayerNorm):
            if hasattr(m, 'activation_quant'):
                if m.activation_quant.activation_n_bits == 8:
                    skip_rr = True
    # input
    with torch.no_grad():
        _ = q_model(calib_data)

    # remove hook
    for h in hooks:
        h.remove()

    folder_path = f"./fp_output/{args.model_name}-calib{args.ridge_regression_num_samples}-W{args.weight_n_bits}A{args.activation_n_bits}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(folder_path, exist_ok=True)
    skip_rr = False
    for n, m in q_model.named_modules():
        if isinstance(m, CompressionWrapper):
            if 'reduction' in n:
                print(f"Skipping reduction layer in RR: {n}")
            else:
                if skip_rr:
                    print(f"Skipping RR in layer: {n}")
                    skip_rr = False
                else:
                    with open(os.path.join(folder_path, n + 'store_input'), 'wb') as file:
                        pickle.dump(m.store_input, file)
                    m.store_input.clear()
                    with open(os.path.join(folder_path, n + 'store_output'), 'wb') as file:
                        pickle.dump(m.store_output, file)
                    m.store_output.clear()
        elif isinstance(m, LayerNorm):
            if hasattr(m, 'activation_quant'):
                if m.activation_quant.activation_n_bits == 8:
                    skip_rr = True
    print("complete collecting fp act...")
    return folder_path


@torch.no_grad()
def replace_W(q_model, folder_path):
    last_act = []
    skip_rr = False
    for n, m in q_model.named_modules():
        if isinstance(m, ActivationQuantizer):
            last_act.append(m)
        if isinstance(m, CompressionWrapper):
            if not m.is_conv:
                if 'reduction' in n:
                    print(f"Skipping reduction layer in RR: {n}")
                    continue
                if skip_rr:
                    print(f"Skipping RR in layer: {n}")
                    skip_rr = False
                    continue
                with open(os.path.join(folder_path, n + 'store_input'), 'rb') as file:
                    store_input = pickle.load(file)
                with open(os.path.join(folder_path, n + 'store_output'), 'rb') as file:
                    store_output = pickle.load(file)

                print("complete collecting act...")
                fp_input = store_input[0]
                if len(fp_input.shape) == 2:
                    num_of_inverse = 0.1
                    print('num_of_inverse', num_of_inverse)
                else:
                    num_of_inverse = 1e-1 * 20000  # coe
                    print('num_of_inverse', num_of_inverse)

                fp_output_shape = store_output[0].shape
                fp_output_flat = store_output[0].to(PRIMARY_DEVICE).reshape(-1, fp_output_shape[-1])
                quan_output = last_act[-1](store_input[0].to(PRIMARY_DEVICE))
                del store_input

                w = m.weight.clone()

                if getattr(m, "bias") is not None:
                    print('bias!')
                    b = m.bias.clone()
                    W_cat = torch.cat((w, b.unsqueeze(1)), dim=1).to(PRIMARY_DEVICE)

                    quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])
                    quan_output_cat = torch.cat((quan_output_flat, torch.ones(quan_output_flat.shape[0], 1).to(PRIMARY_DEVICE)),
                                                dim=1)

                    A = quan_output_cat
                    Y = fp_output_flat - (quan_output_cat @ W_cat.T)

                    beta = torch.inverse(A.permute(1, 0) @ A
                                         + torch.eye(A.shape[1]).to(PRIMARY_DEVICE) * num_of_inverse) @ A.permute(1, 0) @ Y

                    new_W, new_b_0 = torch.split(beta, [beta.shape[0] - 1, 1], dim=0)  # split on the output channel
                    new_b = new_b_0.squeeze()
                    m.weight.data = new_W.T + w
                    m.bias.data = new_b + b

                    del fp_output_flat, quan_output, w, b, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                        beta, new_W, new_b_0, new_b
                    if PRIMARY_DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()

                else:
                    print('None bias!')
                    W_cat = w.to(PRIMARY_DEVICE)
                    quan_output_flat = quan_output.reshape(-1, quan_output.shape[-1])
                    quan_output_cat = quan_output_flat

                    A = quan_output_cat
                    Y = fp_output_flat - (quan_output_cat @ W_cat.T)

                    beta = torch.inverse(A.permute(1, 0) @ A
                                         + torch.eye(A.shape[1]).to(PRIMARY_DEVICE) * num_of_inverse) @ A.permute(1, 0) @ Y

                    new_W = beta
                    m.weight.data = new_W.T + w

                    del fp_output_flat, quan_output, w, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                        beta, new_W
                    if PRIMARY_DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                print(f'complete computing for W in {n}')
                print()
            else:
                if 'embed' in n:
                    print('skip QuantConv2d!')
                    continue
                if 'reduction' in n:
                    print(f"Skipping reduction layer in RR: {n}")
                    continue
                if skip_rr:
                    print(f"Skipping RR in layer: {n}")
                    skip_rr = False
                    continue
                print(n)
                with open(os.path.join(folder_path, n + 'store_input'), 'rb') as file:
                    store_input = pickle.load(file)
                with open(os.path.join(folder_path, n + 'store_output'), 'rb') as file:
                    store_output = pickle.load(file)

                print("complete collecting act...")
                quan_output = last_act[-1](store_input[0].to(PRIMARY_DEVICE))

                kernel_size = m.weight.shape[2]
                stride = m.stride[0]
                padding = m.padding[0]
                quan_output_cols = im2col(quan_output, kernel_size, stride, padding)

                weights_col = deepcopy(m.weight.reshape(m.weight.shape[0], -1).T)

                del store_input

                num_of_inverse = 1e-1 * 20000  # coe
                print('num_of_inverse', num_of_inverse)

                with torch.no_grad():
                    w = weights_col

                    if getattr(m, "bias") is not None:
                        print('bias!')

                        b = m.bias.clone()
                        W_cat = torch.cat((w, b.unsqueeze(0)), dim=0).to(PRIMARY_DEVICE)

                        quan_output_flat = quan_output_cols
                        quan_output_cat = torch.cat((quan_output_flat,
                                                     torch.ones(quan_output_flat.shape[0], 1).to(PRIMARY_DEVICE)),
                                                    dim=1)

                        A = quan_output_cat
                        tmp = (quan_output_cat @ W_cat)
                        fp_output_flat = store_output[0].to(PRIMARY_DEVICE)
                        fp_output_flat = fp_output_flat.permute(0, 2, 3, 1).reshape(tmp.shape)
                        Y = fp_output_flat - tmp
                        beta = torch.inverse(A.permute(1, 0) @ A
                                             + torch.eye(A.shape[1]).to(PRIMARY_DEVICE) * num_of_inverse) @ A.permute(1, 0) @ Y

                        new_W, new_b_0 = torch.split(beta, [beta.shape[0] - 1, 1], dim=0)  # split on the output channel
                        new_b = new_b_0.squeeze()

                        m.weight.data = (new_W + w).T.reshape(m.weight.shape)
                        m.bias.data = new_b + b

                        del fp_output_flat, quan_output, w, b, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                            beta, new_W, new_b_0, new_b
                        if PRIMARY_DEVICE.type == 'cuda':
                            torch.cuda.empty_cache()
                    else:
                        print('None bias!')
                        W_cat = w.to(PRIMARY_DEVICE)

                        quan_output_flat = quan_output_cols

                        A = quan_output_cat
                        tmp = (quan_output_cat @ W_cat)
                        fp_output_flat = store_output[0].to(PRIMARY_DEVICE)
                        fp_output_flat = fp_output_flat.permute(0, 2, 3, 1).reshape(tmp.shape)
                        Y = fp_output_flat - tmp
                        beta = torch.inverse(A.permute(1, 0) @ A
                                             + torch.eye(A.shape[1]).to(PRIMARY_DEVICE) * num_of_inverse) @ A.permute(1, 0) @ Y

                        new_W = beta

                        m.weight.data = (new_W + w).T.reshape(m.weight.shape)

                        del fp_output_flat, quan_output, w, W_cat, quan_output_flat, quan_output_cat, A, Y, \
                            beta, new_W
                        if PRIMARY_DEVICE.type == 'cuda':
                            torch.cuda.empty_cache()
                print(f'complete computing for W in {n}')
                print()

        elif isinstance(m, LayerNorm):
            if hasattr(m, 'activation_quant'):
                if m.activation_quant.activation_n_bits == 8:
                    skip_rr = True
    shutil.rmtree(folder_path)
    return


def ln_reparameterization(model, cc):
    print('Performing scale reparameterization ...')
    with torch.no_grad():
        module_dict = dict(model.named_modules())
        for name, module in model.named_modules():
            if isinstance(module, ChannelWiseActivationQuantizer):
                ln_module = module_dict[module.input_node_name]
                linear_module = module_dict[module.output_node_name]

                act_delta = module.scale.reshape(-1)
                act_zero_point = module.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta

                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                ln_module.weight.data.copy_(ln_module.weight.data / r)
                ln_module.bias.data.copy_(ln_module.bias.data / r - b)

                linear_module.weight.data.copy_(linear_module.weight.data * r)

                if hasattr(linear_module, 'bias'):
                    if linear_module.bias is not None:
                        linear_module.bias.data.copy_(linear_module.bias.data
                                                      + torch.mm(linear_module.weight.data,
                                                                 b.reshape(-1, 1)).reshape(-1))
                    else:
                        linear_module.bias = nn.Parameter(torch.ones(linear_module.out_features,
                                                                     device=linear_module.weight.device))
                        linear_module.bias.data.copy_(torch.mm(linear_module.weight.data, b.reshape(-1, 1)).reshape(-1))
                else:
                    if linear_module.base_module.bias is not None:
                        linear_module.base_module.bias.data.copy_(linear_module.base_module.bias.data
                                                                  + torch.mm(linear_module.weight.data,
                                                                             b.reshape(-1, 1)).reshape(-1))
                    else:
                        linear_module.base_module.bias = nn.Parameter(torch.ones(linear_module.base_module.out_features,
                                                                                 device=linear_module.weight.device))
                        linear_module.base_module.bias.data.copy_(
                            torch.mm(linear_module.weight.data, b.reshape(-1, 1)).reshape(-1))
                linear_module.inited = False
                new_quantizer = ActivationQuantizer(name=module.name, compression_config=cc)
                new_quantizer.scale = nn.Parameter(target_delta)
                new_quantizer.zero_point = target_zero_point
                new_quantizer.activation_n_bits = module.activation_n_bits
                if hasattr(module, 'set_size'):
                    new_quantizer.set_size = module.set_size
                model.add_submodule(module.name, new_quantizer)

    return model


def add_channel_wise_activation_quantizer(node, activation_quantizer, compression_config, graph, model, module=None,
                                          input_node_name=None, output_node_name=None):
    if node.op == CALL_MODULE:
        node_name = node.target
    else:
        node_name = node.name

    # Generate unique name for the quantizer
    quant_module_name = f"{node_name}.{ACTIVATION_QUANT_STR}"
    # Create ActivationQuantizer
    if activation_quantizer == ChannelWiseActivationQuantizer:
        quantizer = activation_quantizer(quant_module_name, compression_config, module, input_node_name,
                                         output_node_name)
    else:
        quantizer = activation_quantizer(quant_module_name, compression_config)
    # Add quantizer to submodules
    model.add_submodule(quant_module_name, quantizer)
    replace_input_with(graph, node, quant_module_name)
    return model


def log2sqrt_quantization(x, delta, act_nbits):
    x_int = torch.round(-1 * (x / delta).log2() * 2)
    mask = x_int >= 2 ** act_nbits
    x_quant = torch.clamp(x_int, 0, 2 ** act_nbits - 1)
    odd_mask = (x_quant % 2) * (sqrt(2) - 1) + 1
    x_float_q = 2 ** (-1 * torch.ceil(x_quant / 2)) * odd_mask * delta
    x_float_q[mask] = 0

    return x_float_q


def log2sqrt_search_activation_params(x, act_nbits):
    delta = None
    x_clone = x.clone().detach()
    delta = x_clone.max()
    best_score = 1e+10
    for pct in [0.999, 0.9999, 0.99999]:
        try:
            new_delta = torch.quantile(x_clone.reshape(-1), pct)
        except:
            new_delta = torch.tensor(np.percentile(
                x_clone.reshape(-1).cpu(), pct * 100),
                device=x_clone.device,
                dtype=torch.float32)
        x_q = log2sqrt_quantization(x_clone, new_delta, act_nbits)
        score = lp_loss(x_clone, x_q, p=2, reduction='all')
        if score < best_score:
            best_score = score
            delta = new_delta

    return delta


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def uniform_quantization(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def uniform_quantization_min_max(x, max_level, min_level, n_bits):
    delta = (max_level - min_level) / (2 ** n_bits - 1)
    zero_point = (- min_level / delta).round()
    # we assume weight quantization is always signed
    x_int = torch.round(x / delta)
    x_quant = torch.clamp(x_int + zero_point, 0, 2 ** n_bits - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q


def search_activation_params(x: torch.Tensor,
                             act_nbits: int,
                             mse_p: float = 2.0,
                             quantize_function: Callable = uniform_quantization_min_max):
    delta, zero_point = None, None
    x_clone = x.clone().detach()
    best_score = 1e+10

    for pct in [0.999, 0.9999, 0.99999]:
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
        x_q = quantize_function(x_clone, new_max, new_min, act_nbits)
        score = ((x_q - x).abs().pow(mse_p)).mean()  # lp_loss(x_clone, x_q, p=2, reduction='all')
        if score < best_score:
            best_score = score
            delta = (new_max - new_min) / (2 ** act_nbits - 1)
            zero_point = (- new_min / delta).round()

    return delta, zero_point


def search_channel_wise_activation_params(x: torch.Tensor,
                                          act_nbits: int,
                                          mse_p: float = 2.0,
                                          quantize_function: Callable = uniform_quantization_min_max):
    delta, zero_point = None, None
    x_clone = x.clone().detach()
    n_channels = x_clone.shape[-1]
    if len(x.shape) == 4:
        # x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0].max(dim=0)[0]
        # x_max = x_clone.abs().max(dim=0)[0].max(dim=-1)[0].max(dim=-1)[0]
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
        if len(x.shape) == 3:
            delta[c], zero_point[c] = search_activation_params(x_clone[:, :, c], act_nbits=act_nbits)
        else:
            delta[c], zero_point[c] = search_activation_params(x_clone[:, :, :, c], act_nbits=act_nbits)
    if len(x.shape) == 4:
        delta = delta.view(1, 1, 1, -1)
        zero_point = zero_point.view(1, 1, 1, -1)
    elif len(x.shape) == 2:
        delta = delta.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    elif len(x.shape) == 3:
        delta = delta.view(1, 1, -1)
        zero_point = zero_point.view(1, 1, -1)
    else:
        raise NotImplementedError
    return delta, zero_point


class ActivationQuantizer(nn.Module):
    def __init__(self, name, compression_config: CompressionConfig):
        super().__init__()
        self.name = name
        self.compression_config = compression_config

        self.activation_n_bits = self.compression_config.activation_n_bits
        self.activation_quantization = False
        self.a_qmin, self.a_qmax = None, None

        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float).to(DEVICE))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int).to(DEVICE))

        self.search_qparams = False

    def set_layer_qparams_search(self, v, set_size):
        self.search_qparams = v
        self.set_size = set_size

    def set_activation_quantization(self, v: bool):
        self.activation_quantization = v

    def forward(self, x):
        if self.set_size:
            self.base_act_size = int(np.prod(list(x.shape[1:])))
        if self.search_qparams:
            _scale, _zero_point = search_activation_params(
                x=x,
                act_nbits=self.activation_n_bits)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        if self.activation_quantization:
            x = uniform_quantization(x, self.scale, self.zero_point.item(), 0, 2 ** self.activation_n_bits - 1)
        return x


class ChannelWiseActivationQuantizer(ActivationQuantizer):
    def __init__(self, name: str, compression_config: CompressionConfig, module, input_node_name, output_node_name):
        super().__init__(name, compression_config)

        self.scale = torch.nn.Parameter(torch.ones([1, 1, module.weight.shape[0]], dtype=torch.float).to(DEVICE))
        self.register_buffer('zero_point', torch.zeros([1, 1, module.weight.shape[0]], dtype=torch.float).to(DEVICE))

        self.input_node_name = input_node_name
        self.output_node_name = output_node_name

    def forward(self, x):
        if self.set_size:
            self.base_act_size = int(np.prod(list(x.shape[1:])))
        if self.search_qparams:
            _scale, _zero_point = search_channel_wise_activation_params(
                x=x,
                act_nbits=self.activation_n_bits)
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            if len(x.shape) == 4:
                self.scale = torch.nn.Parameter(
                    torch.ones(_scale.shape, dtype=torch.float).to(DEVICE))
                self.zero_point = torch.nn.Parameter(
                    torch.ones(_zero_point.shape, dtype=torch.float).to(DEVICE))
            self.scale.data.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        if self.activation_quantization:
            x = uniform_quantization(x, self.scale, self.zero_point, 0, 2 ** self.activation_n_bits - 1)
        return x


class LogSqrt2ActivationQuantizer(ActivationQuantizer):
    def __init__(self, name: str, compression_config: CompressionConfig):
        super().__init__(name, compression_config)

    def forward(self, x):
        if self.set_size:
            self.base_act_size = int(np.prod(list(x.shape[1:])))
        if self.search_qparams:
            _scale = log2sqrt_search_activation_params(
                x=x,
                act_nbits=self.activation_n_bits)
            _scale = _scale.to(self.scale.device)
            self.scale.data.copy_(_scale)
        if self.activation_quantization:
            x = log2sqrt_quantization(x, self.scale, self.activation_n_bits)
        return x


def add_activation_quantizer(node, activation_quantizer, compression_config, graph, model, next_node):
    if next_node.op == CALL_MODULE:
        next_node_name = next_node.target
    else:
        next_node_name = next_node.name

    modules_names = dict(model.named_modules()).keys()

    # Generate unique name for the quantizer
    quant_module_name = f"{next_node_name}.{ACTIVATION_QUANT_STR}"

    if quant_module_name in modules_names:
        count = sum(quant_module_name in name for name in modules_names)
        quant_module_name = quant_module_name + '_' + str(count)

    # Create ActivationQuantizer
    quantizer = activation_quantizer(quant_module_name, compression_config)
    # Add quantizer to submodules
    model.add_submodule(quant_module_name, quantizer)
    replace_input_with(graph, node, quant_module_name)
    return model


def replace_input_with(graph: torch.fx.Graph,
                       input_node: Node,
                       new_node_name: str,
                       ) -> Node:
    users = list(input_node.users)
    for user in users:
        with graph.inserting_after(input_node):
            new_node = graph.call_module(new_node_name, args=(input_node,))
        user.replace_input_with(input_node, new_node)
    return new_node


def remove_nodes(model, graph, modules, nodes_remove=REDUNDANT_OPERATORS):
    modules_to_remove = []
    for node in list(graph.nodes):
        if node.op == CALL_MODULE:
            module = modules[node.target]
            if type(module) in nodes_remove:
                # Save the input node and dependent nodes
                input_node = node.args[0]  # Input to the node being removed
                dependent_nodes = list(node.users)  # Nodes that consume this node's output

                # Update dependent nodes to use the input node instead
                for dependent in dependent_nodes:
                    dependent.replace_input_with(node, input_node)

                # Erase the current node from the graph
                graph.erase_node(node)
                modules_to_remove.append(node.target)
    # Find and remove the module from the model
    for name, module in list(model.named_modules()):
        if name in modules_to_remove:
            model.delete_submodule(name)
    return model, list(graph.nodes)


def set_model_activation_qparams_search(model, v, set_size=False):
    for n, m in model.named_modules():
        if isinstance(m, ActivationQuantizer):
            m.set_layer_qparams_search(v, set_size)


def set_model_activation_bit_width(quant_model,
                                   calib_samples,
                                   model_manager,
                                   compression_config):
    set_model_activation_qparams_search(quant_model, False, set_size=True)
    with torch.no_grad():
        model_manager.forward(quant_model, calib_samples)

    size_dict = {}
    for n, m in quant_model.named_modules():
        if isinstance(m, ActivationQuantizer):
            size_dict.update({n: m.base_act_size})

    current_max_size = FLOAT_BIT_WIDTH * max(size_dict.values())
    target_max_size = current_max_size * compression_config.activation_n_bits / FLOAT_BIT_WIDTH
    options = np.asarray(MP_BIT_WIDTH)
    bit_width_solve = {}

    print("Activation MP bits solution:")
    for n, size in size_dict.items():
        options_size = size * options
        ind = options_size <= target_max_size
        bit_sel = np.max(np.where(ind))
        bit_width_solve.update({n: options[bit_sel]})
        print(f"{n}:\t {options[bit_sel]:2.3f}")

    for n, m in quant_model.named_modules():
        if isinstance(m, ActivationQuantizer):
            bit_width = bit_width_solve.get(n)
            m.activation_n_bits = bit_width
    return quant_model


def remove_channelwise_quantizers(model, cc):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, ChannelWiseActivationQuantizer):
                if module.activation_n_bits == 8:
                    new_quantizer = ActivationQuantizer(name=module.name, compression_config=cc)
                    new_quantizer.activation_n_bits = module.activation_n_bits
                    if hasattr(module, 'set_size'):
                        new_quantizer.set_size = module.set_size

                    model.add_submodule(module.name, new_quantizer)

    return model


def activation_quantization_param_search(quant_model,
                                         calib_samples,
                                         model_manager,
                                         compression_config):
    if compression_config.activation_mp:
        quant_model = set_model_activation_bit_width(quant_model, calib_samples, model_manager, compression_config)

    quant_model = remove_channelwise_quantizers(quant_model, cc=compression_config)

    set_model_activation_qparams_search(quant_model, True)

    with torch.no_grad():
        model_manager.forward(quant_model, calib_samples)

    set_model_activation_qparams_search(quant_model, False)

    print('Activation parameter search done.')
    return quant_model


def insert_activation_quantization_to_graph(
        model: Module,
        graph: Graph,
        nodes: List,
        modules: Dict,
        input_activations_quant: List,
        weight_quantizer_wrapper: Type[Module],
        activation_quantizer: Type[Module],
        compression_config: CompressionConfig,
):
    wrapper_nodes = []
    for node in nodes:
        if node.op == CALL_MODULE:
            module = modules[node.target]
            if isinstance(module, weight_quantizer_wrapper):
                wrapper_nodes.append(node)

    nodes_for_channel_reparam = []

    if not compression_config.disable_ln_reparam:
        nodes_for_channel_reparam.append(LayerNorm)

    nodes_for_log_scale = []

    if not compression_config.disable_softmax_log_scale:
        nodes_for_log_scale.append('softmax')
        nodes_for_log_scale.append(torch.nn.modules.activation.Softmax)

    next_weight_quantizer_wrapper = None
    for node in nodes:
        if node.op == CALL_MODULE:
            module = modules[node.target]
            if isinstance(module, weight_quantizer_wrapper):
                module = module.base_module
                next_weight_quantizer_wrapper = node
            activation_type = type(module)
        elif node.op == CALL_FUNCTION:
            activation_type = node.target
        elif node.op == CALL_METHOD:
            activation_type = node.target
        else:
            continue  # Skip other types of nodes

        if activation_type in [Conv2d]:
            continue
        if activation_type in input_activations_quant:
            if next_weight_quantizer_wrapper == node:
                next_weight_quantizer_wrapper_for_name = wrapper_nodes[
                    wrapper_nodes.index(next_weight_quantizer_wrapper) - 1]
            else:
                next_weight_quantizer_wrapper_for_name = next_weight_quantizer_wrapper
            for input_node in node.all_input_nodes:
                if input_node.op == CALL_MODULE:
                    module = modules[input_node.target]
                    activation_type = type(module)
                elif input_node.op == CALL_METHOD:
                    activation_type = input_node.target
                elif input_node.op == CALL_FUNCTION:
                    activation_type = input_node.target

                if activation_type in nodes_for_log_scale:
                    model = add_activation_quantizer(input_node, LogSqrt2ActivationQuantizer, compression_config, graph,
                                                     model, next_node=next_weight_quantizer_wrapper_for_name)
                elif activation_type in nodes_for_channel_reparam:
                    if 'reduction' not in node.target:
                        model = add_channel_wise_activation_quantizer(input_node, ChannelWiseActivationQuantizer,
                                                                      compression_config, graph,
                                                                      model, modules[input_node.target],
                                                                      input_node.target, node.target)
                    else:
                        model = add_activation_quantizer(input_node, activation_quantizer, compression_config,
                                                         graph, model,
                                                         next_node=next_weight_quantizer_wrapper_for_name)
                        print(f"Skipping reduction layer in LN reparam: {node.target}")
                else:
                    model = add_activation_quantizer(input_node, activation_quantizer, compression_config, graph, model,
                                                     next_node=next_weight_quantizer_wrapper_for_name)

    return model, graph


def insert_activation_quantization(model: Module,
                                   input_activations_quant: list,
                                   compression_config,
                                   weight_quantizer_wrapper=CompressionWrapper,
                                   activation_quantizer=ActivationQuantizer):
    graph = model.graph
    modules = dict(model.named_modules())

    # Remove redundant nodes
    model, nodes = remove_nodes(model=model, graph=graph, modules=modules)
    modules = dict(model.named_modules())

    # Now, insert ActivationQuantizer after every operator not in any pattern
    model, graph = insert_activation_quantization_to_graph(model=model, graph=graph, nodes=nodes, modules=modules,
                                                           input_activations_quant=input_activations_quant,
                                                           weight_quantizer_wrapper=weight_quantizer_wrapper,
                                                           activation_quantizer=activation_quantizer,
                                                           compression_config=compression_config)

    graph.lint()
    model.recompile()
    return model
