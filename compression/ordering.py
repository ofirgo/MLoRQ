import torch.linalg
from tqdm import tqdm

from compression.configs.compression_config import CompressionConfig, ParetoCost
from compression.quantization.quantization import uniform_quantization
from utils.candidates_utils import get_quantization_only_candidates, get_sorted_candidates_per_ab_bit


def compute_decomposed_weights_mse_cost(in_a, in_b, rank, lcc, cost):
    Ar = in_a[:, :rank]
    Br = in_b[:rank, :]

    Ar_q = lcc.quantize_Ar_uniform(Ar, rank)
    Br_q = lcc.quantize_Br_uniform(Br, rank)

    w_lr = Ar_q @ Br_q
    mse_cost_lrq = cost(w_lr)
    return mse_cost_lrq


def run_pareto(in_list_to_pareto):
    _pareto = torch.tensor([p[:2] for p in in_list_to_pareto])
    mse_array = _pareto[:, 0]
    size_array = _pareto[:, 1]

    N = mse_array.size(0)
    mse_array_exp = mse_array.unsqueeze(0).expand(N, N)
    size_array_exp = size_array.unsqueeze(0).expand(N, N)

    # Check for domination conditions
    mse_array_dominated = (mse_array_exp <= mse_array_exp.T)
    size_array_dominated = (size_array_exp <= size_array_exp.T)

    # Exclude self-comparisons
    self_mask = torch.eye(N, dtype=torch.bool, device=mse_array.device)

    # A point is dominated if there exists another point satisfying both conditions
    dominated = torch.any((mse_array_dominated & size_array_dominated) & ~self_mask, dim=1)

    # Select undominated points
    undominated_mask = ~dominated

    res = [in_list_to_pareto[i] for i in range(len(undominated_mask)) if undominated_mask[i]]
    return res


def generate_pareto_cost(weighting, in_w, cost_type: ParetoCost):
    def _cost(w_tilde):

        error = in_w - w_tilde
        if cost_type == ParetoCost.MSE:
            return torch.mean(error ** 2)
        elif cost_type == ParetoCost.HMSEPerOutChannel:
            if len(weighting.shape) == 4:
                w_per_out_channel = torch.mean(weighting, dim=[-1, -2]).max(dim=-1, keepdim=True)[0]
            else:
                w_per_out_channel = torch.max(weighting, dim=-1, keepdim=True)[0]
            return torch.mean(w_per_out_channel * error ** 2)  # Ordering score
        else:
            raise NotImplemented
    return _cost


def generate_point_ordering(in_weights, compression_options, base_size, n_in, n_out, in_a, in_b,
                            in_cc: CompressionConfig, hessian_for_pareto):

    local_pareto_scores = []
    cost = generate_pareto_cost(hessian_for_pareto, in_weights, in_cc.pareto_cost)



    # Compute MSE cost for quantization only candidates
    quantization_only_candidates = get_quantization_only_candidates(in_cc.weight_bit_list, compression_options)
    for _, lcc in quantization_only_candidates.items():
        w_q = uniform_quantization(in_weights,
                                   lcc.delta.reshape((lcc.delta.shape[0], -1)),
                                   lcc.zero_point.reshape((lcc.zero_point.shape[0], -1)),
                                   2 ** lcc.bit_width_quantization)
        cost_q = cost(w_q).item()
        size_q = base_size * lcc.bit_width_quantization
        local_pareto_scores.append([cost_q, size_q, lcc])

    # Compute MSE cost for LR candidates
    sorted_lr_cfg_per_bitwidth = get_sorted_candidates_per_ab_bit(in_cc.weight_bit_list, compression_options)
    for _, cfgs_list in tqdm(sorted_lr_cfg_per_bitwidth.items(), "Computing local scores for pareto..."):
        for lcc in cfgs_list:
            cost_q = compute_decomposed_weights_mse_cost(in_a, in_b, lcc.rank, lcc, cost).item()
            size_q = lcc.size(n_in, n_out).item()
            local_pareto_scores.append([cost_q, size_q, lcc])

    # Compute Pareto front
    pareto = run_pareto(local_pareto_scores)
    pareto = [p for p in pareto if p[2].rank is None or p[2].rank > 1]
    x = [p for p in local_pareto_scores if p[2].rank is None and p not in pareto]
    pareto += x

    return pareto