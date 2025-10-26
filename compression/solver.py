from typing import Dict, Optional

import torch
from ortools.constraint_solver.pywrapcp import IntVar, Solver
from ortools.linear_solver import pywraplp
from tqdm import tqdm

from compression.error_interpolation import compute_sqnr_interpolation_score, compute_batch_score_per_config, \
    normalize_per_batch_score, ScorePoint
from constants import SOLVER_TIME_LIMIT
from helpers.utils import is_compressed_layer


def init_lp_vars(solver: Solver, layer_to_metrics_mapping: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, IntVar]]:
    layer_to_indicator_vars_mapping = dict()

    for i, (layer, nbits_to_metric) in enumerate(layer_to_metrics_mapping.items()):
        layer_to_indicator_vars_mapping[layer] = dict()

        for nbits in nbits_to_metric.keys():
            var = solver.IntVar(0, 1, f"layer_{i}_{layer}_{nbits}")
            layer_to_indicator_vars_mapping[layer][nbits] = var

    return layer_to_indicator_vars_mapping


def formalize_problem(solver,
                      layer_to_indicator_vars_mapping: Dict[str, Dict[int, IntVar]],
                      layer_to_metrics_mapping: Dict[str, Dict[int, float]],
                      layer_to_size_mapping: Optional[Dict[str, Dict[int, float]]],
                      target_weights_memory: Optional[float]):
    layers = layer_to_metrics_mapping.keys()

    # Objective (minimize acc loss)
    objective = solver.Objective()

    # Objective (minimize loss)
    for layer in layers:
        for nbits, indicator in layer_to_indicator_vars_mapping[layer].items():
            objective.SetCoefficient(indicator, layer_to_metrics_mapping[layer][nbits])
    objective.SetMinimization()

    # Constraint of only one indicator == 1 for each layer
    for layer in layers:
        constraint = solver.Constraint(1, 1)  # Enforcing that sum == 1
        for v in layer_to_indicator_vars_mapping[layer].values():
            constraint.SetCoefficient(v, 1)

    # Constraint for model size
    if target_weights_memory is not None:
        weights_memory_constraint = solver.Constraint(-solver.infinity(), target_weights_memory)
    else:
        weights_memory_constraint = None

    for layer in layers:
        for nbits_idx, indicator in layer_to_indicator_vars_mapping[layer].items():
            if target_weights_memory:
                weights_memory_constraint.SetCoefficient(indicator, layer_to_size_mapping[layer][nbits_idx])

    return solver


def _build_index2name_mapping(qm):
    index = 0
    index2name = {}
    ##############################################################################
    # Scan model and disable compression
    ##############################################################################
    for n, m in tqdm(qm.named_modules(), desc='Apply quantization to modules'):
        if is_compressed_layer(m):
            index2name.update({index: (n, m)})
            index += 1

    return index2name


def _compute_quant_float_sizes(compressed_layers_index2name):
    size_map = {layer_index: {} for layer_index in compressed_layers_index2name}
    float_size = 0

    for layer_index, (n, m) in compressed_layers_index2name.items():
        float_size += m.compute_float_size()
        for config_index, compression_config in enumerate(m.pareto_config):
            size_map[layer_index][config_index] = m.compute_size(cfg=compression_config)

    return size_map, float_size


def build_ips_distance_mapping(compressed_layers_index2name, cc, representative_data_loader,
                               model_manager, output_ref, qm):
    distance_map = {layer_index: {} for layer_index in compressed_layers_index2name}

    # compute global error scores
    for layer_index, (n, m) in tqdm(compressed_layers_index2name.items(),
                                    "Computing per-layer global error for Mixed Precision..."):

        ## Quant only scores
        quant_only_cfgs = {ScorePoint(sorted_lr_idx=None, pareto_idx=i): cfg for i, cfg in
                           enumerate(m.pareto_config) if cfg.rank is None}
        layer_quant_only_scores = compute_batch_score_per_config(quant_only_cfgs, representative_data_loader, model_manager, output_ref, m, qm)
        layer_quant_only_scores = normalize_per_batch_score(layer_quant_only_scores, m.weight.device)

        ## LR scores using interpolation
        layer_lr_inter_scores = compute_sqnr_interpolation_score(m, cc, representative_data_loader, model_manager,
                                                                 output_ref, qm)

        distance_map[layer_index].update({k.pareto_idx: score for k, score in layer_quant_only_scores.items()})
        distance_map[layer_index].update(layer_lr_inter_scores)

    return distance_map


def run_solver(qmodel, cc, representative_data_loader, model_manager, output_ref):
    with torch.no_grad():
        index2nm = _build_index2name_mapping(qmodel)
        layer_to_size_mapping, float_size = _compute_quant_float_sizes(index2nm)

        print(f"Model Float Size [bytes]: {float_size}")

        layer_to_metrics_mapping = build_ips_distance_mapping(index2nm, cc, representative_data_loader, model_manager,
                                                              output_ref, qmodel)

        print(layer_to_metrics_mapping)

    def opt_func(target_compression_rate):
        solver = pywraplp.Solver.CreateSolver('CBC')  # Use 'CBC' as solver
        layer_to_indicator_vars_mapping = init_lp_vars(solver, layer_to_metrics_mapping)
        target_weights_memory = target_compression_rate * float_size
        lp_problem = formalize_problem(solver,
                                       layer_to_indicator_vars_mapping=layer_to_indicator_vars_mapping,
                                       layer_to_metrics_mapping=layer_to_metrics_mapping,
                                       layer_to_size_mapping=layer_to_size_mapping,
                                       target_weights_memory=target_weights_memory)

        # Solve the problem with a time limit (if applicable)
        lp_problem.SetTimeLimit(SOLVER_TIME_LIMIT * 1000)  # OR-Tools time limit is in milliseconds
        status = lp_problem.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print("Solution found!")
        elif status == pywraplp.Solver.FEASIBLE:
            print("Feasible solution found within time limit.")
        else:
            raise Exception("No solution found.")

        # Retrieve the solution values for the indicators
        indicators_values_per_layer = [
            layer_to_indicator_vars_mapping[layer_idx]
            for layer_idx, (name, m) in index2nm.items()
        ]

        # Ensure exactly one indicator variable is set to 1 per layer
        for layer_indicators in indicators_values_per_layer:
            assert sum(indicator.solution_value() for indicator in layer_indicators.values()) == 1, \
                "ILP solution should include exactly one candidate with indicator value 1 for each layer."

        # Collect compression results for each layer
        compression_results = {
            name: m.pareto_config[
                next(b_idx for b_idx, ind in indicators_values_per_layer[layer_index].items() if
                     ind.solution_value() == 1)
            ]
            for layer_index, (name, m) in index2nm.items()
        }

        res = []
        for k, v in compression_results.items():
            if v.is_low_rank:
                res.append((k, v.bit_width_quantization_a, v.bit_width_quantization_b, v.rank))
            else:
                res.append((k, v.bit_width_quantization,))
        print(res)
        return compression_results

    return opt_func
