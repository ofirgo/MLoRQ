"""
 Some functions in this file is copied from https://github.com/pytorch/pytorch
 and modified for this project's needs.
"""

from model_managers.base_model import BaseModel

import copy
from typing import Dict, Iterable, Type, Any, Tuple, Callable

import torch
from torch import fx
from tqdm import tqdm


def _parent_name(target: str) -> Tuple[str, str]:
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def _replace_node_module(node: fx.Node, modules: Dict[str, Any], replace_function: Callable):
    assert (isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    new_module = replace_function(modules[node.target], node_name=node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)
    return new_module


def _matches_module_pattern(pattern: Iterable[Type], node: fx.Node, in_node: fx.Node, modules: Dict[str, Any]):
    nodes: Tuple[Any, fx.Node] = (in_node, node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace2quantized_model(in_model: torch.nn.Module, model_manager: BaseModel, replace_function: Callable,
                            linear_patterns, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """

    if not inplace:
        in_model = copy.deepcopy(in_model)
    fx_model = model_manager.get_fx_graph(in_model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in linear_patterns:
        for node in tqdm(new_graph.nodes):
            for in_node in node.all_input_nodes:
                if _matches_module_pattern(pattern, node, in_node, modules):
                    should_skip = any([skip_layer_name in in_node.name for skip_layer_name in model_manager.skip_quantization_layer_names])
                    if should_skip:
                        continue
                    target_op = modules[in_node.target]

                    wrap_node = _replace_node_module(in_node, modules,
                                                     replace_function
                                                     )

    return fx.GraphModule(fx_model, new_graph)
