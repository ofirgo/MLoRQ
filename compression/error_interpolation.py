from itertools import product
from typing import NamedTuple, Union

import torch
from compression.quantization.distance_metrics import sqnr_distance

"""
sorted_lr_idx: the index of the config option in a sorted per-ab-bitwidth options list
pareto_idx: the index of the config in the module's pareto_config list
"""
ScorePoint = NamedTuple('ScorePoint', [('sorted_lr_idx', Union[int, None]), ('pareto_idx', int)])


def compute_batch_score_per_config(base_compression_options, representative_data_loader, model_manager, output_ref,
                                   compression_wrapper_module, qm):
    """
    base_compression_options is ScorePoint --> config
    """

    batch_scores_per_cfg = {k: [] for k in base_compression_options}

    with torch.no_grad():
        compression_wrapper_module.enable_compression()
        for batch_idx, batch in enumerate(representative_data_loader):
            data = model_manager.data_to_device(batch)
            batch_output_ref = output_ref[batch_idx].to(model_manager.device)
            for k, compression_config in base_compression_options.items():
                compression_wrapper_module.set_compression_config(compression_config)  # Set compression config
                score = sqnr_distance(qm, data, batch_output_ref, model_manager)
                batch_scores_per_cfg[k].append(score)
        compression_wrapper_module.disable_compression()

    return batch_scores_per_cfg


def normalize_per_batch_score(batch_scores_per_cfg, device):
    """
    batch_scores_per_cfg is ScorePoint --> List of per batch scores
    """
    return {
        k: torch.mean(torch.tensor([s[0] for s in scores], device=device)).item()
           / torch.mean(torch.tensor([s[1] for s in scores], device=device)).item()
        for k, scores in batch_scores_per_cfg.items()}


def compute_sqnr_for_base_candidates(inter_points, representative_data_loader, model_manager, output_ref,
                                     m, qm):
    batch_scores_per_cfg = compute_batch_score_per_config(inter_points, representative_data_loader,
                                                          model_manager, output_ref, m, qm)

    sqnr_per_base_config = normalize_per_batch_score(batch_scores_per_cfg, m.weight.device)

    return sqnr_per_base_config


def compute_sqnr_for_interpolated_candidates(cfgs_map, m, idx_base_cfg_for_bitwidth, sqnr_per_base_config):
    scores = {}
    sorted_lr_pareto_local_score = [m.pareto[i][0] for i, c in enumerate(m.pareto_config)]
    sqnr_base_scores = {k.sorted_lr_idx: s for k, s in sqnr_per_base_config.items()}
    for k, lcc in cfgs_map.items():
        idx, og_idx = k.sorted_lr_idx, k.pareto_idx
        if (idx, og_idx) in [x for x in idx_base_cfg_for_bitwidth]:
            scores[og_idx] = sqnr_base_scores[idx]
            continue  # base config, sqnr already computed

        (bottom_idx, bottom_idx_og), (top_idx, top_idx_og) = (max([i for i in idx_base_cfg_for_bitwidth if i[0] < idx]),
                                                              min([i for i in idx_base_cfg_for_bitwidth if i[0] > idx]))

        f_r = sorted_lr_pareto_local_score[og_idx]
        f_r_max = sorted_lr_pareto_local_score[top_idx_og]
        f_r_min = sorted_lr_pareto_local_score[bottom_idx_og]
        w_r = (f_r - f_r_max) / (f_r_min - f_r_max)
        cost_q = (sqnr_base_scores[top_idx] * (1 - w_r) + sqnr_base_scores[bottom_idx] * w_r)  # interpolation

        scores[og_idx] = cost_q  # key is index in original pareto config list

    return scores


def compute_sqnr_interpolation_score(m, cc, representative_data_loader, model_manager, output_ref, qm):
    num_inter_points = cc.num_inter_points
    full_per_cfg_score = {}

    for n_a, n_b in list(product(cc.weight_bit_list, cc.weight_bit_list)):
        lr_candidates = [(i, c) for i, c in enumerate(m.pareto_config) if
                         c.rank is not None and c.bit_width_quantization_a == n_a and c.bit_width_quantization_b == n_b]

        if len(lr_candidates) > 0:
            sorted_lr = sorted(lr_candidates, key=lambda t: t[1].rank)
            lr_sorted_to_pareto_sorted = {i: pareto_idx for i, (pareto_idx, _) in enumerate(sorted_lr)}

            ## Compute base points SQNR scores
            if len(sorted_lr) <= num_inter_points:
                inter_boundary_points_idxs = [k for k, v in lr_sorted_to_pareto_sorted.items()]
            else:
                inter_boundary_points_idxs = [round(i * (len(lr_candidates) - 1) / (num_inter_points - 1)) for i in
                                              range(num_inter_points)]

            base_points = {ScorePoint(sorted_lr_idx=i, pareto_idx=lr_sorted_to_pareto_sorted[i]): sorted_lr[i][1] for i
                           in inter_boundary_points_idxs}
            base_points_global_scores = compute_sqnr_for_base_candidates(base_points, representative_data_loader,
                                                                         model_manager, output_ref, m, qm)

            ## Compute interpolation points SQNR scores
            idx_base_cfg_for_bitwidth = [(i, lr_sorted_to_pareto_sorted[i]) for i in inter_boundary_points_idxs]
            all_lr_configs = {ScorePoint(sorted_lr_idx=i, pareto_idx=pareto_idx): c for i, (pareto_idx, c) in
                              enumerate(sorted_lr)}
            per_cfg_scores = compute_sqnr_for_interpolated_candidates(all_lr_configs,
                                                                      m,
                                                                      idx_base_cfg_for_bitwidth,
                                                                      base_points_global_scores)
            full_per_cfg_score.update(per_cfg_scores)

    return {k: v if isinstance(v, float) else v.item() for k, v in full_per_cfg_score.items()}