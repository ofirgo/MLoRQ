from itertools import product
from typing import NamedTuple, Optional


class CandidateKey(NamedTuple):
    w_nbits: Optional[int] = None
    Ar_nbits: Optional[int] = None
    Br_nbits: Optional[int] = None
    rank: Optional[int] = None


def get_quantization_only_candidates(weight_bit_list, compression_options):
    return {CandidateKey(w_nbits=n): compression_options.get_quantization_only_compression(n) for n in weight_bit_list}


def get_sorted_candidates_per_ab_bit(weight_bit_list, compression_options):
    sorted_lr_cfg_per_bitwidth = {}

    for n_a, n_b in list(product(weight_bit_list, weight_bit_list)):
        lr_candidates = compression_options.get_quantization_and_low_rank_options(n_a, n_b)

        if len(lr_candidates) > 0:
            k = CandidateKey(Ar_nbits=n_a, Br_nbits=n_b)
            sorted_lr = sorted(lr_candidates, key=lambda c: c.rank)
            sorted_lr_cfg_per_bitwidth[k] = sorted_lr

    return sorted_lr_cfg_per_bitwidth