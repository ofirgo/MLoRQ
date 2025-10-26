import math
from functools import partial
from itertools import product

import numpy as np
import torch

from compression.configs.layer_comrpression_config import LayerCompressionConfig


class CompressionOptions:
    def __init__(self, in_n_out, in_n_in, in_compression_config, enable_low_rank, device, compression_options=None):
        self.n_out = in_n_out
        self.n_in = in_n_in
        self.base_size = self.n_in * self.n_out
        self.compression_config = in_compression_config

        self.compression_options_list = [LayerCompressionConfig(is_low_rank=False, bit_width_quantization=nbits) for
                                         nbits in
                                         in_compression_config.weight_bit_list] if compression_options is None else \
            compression_options
        self.rank_range_array = torch.tensor(
            np.linspace(1, min(self.n_out, self.n_in), min(self.n_out, self.n_in)).astype("int"), device=device)

        if enable_low_rank and compression_options is None:
            self._build_compression_options()

    def get_compression_options_list(self):
        return self.compression_options_list

    def get_quantization_only_compression(self, bit_width):
        return [c for c in self.compression_options_list if c.rank is None and c.bit_width_quantization == bit_width][0]

    def get_quantization_and_low_rank_options(self, bit_width_a, bit_width_b):
        return [c for c in self.compression_options_list if
                c.rank is not None and c.bit_width_quantization_a == bit_width_a and c.bit_width_quantization_b == bit_width_b]

    def _build_compression_options(self):
        in_bit_widths = self.compression_config.weight_bit_list
        bit_for_lr = [b for b in in_bit_widths]
        # Disable Low-Rank for 2,3 bits.
        if 2 in bit_for_lr:
            bit_for_lr.remove(2)
        if 3 in bit_for_lr:
            bit_for_lr.remove(3)
        A_B_nbits_options = list(product(bit_for_lr, bit_for_lr))

        for na, nb in A_B_nbits_options:
            r_opt = self._get_rank_options_for_bitwidths(na, nb, in_bit_widths, self.base_size,
                                                         partial(self._simd_fn, simd=self.compression_config.simd))
            for r in r_opt:
                self.compression_options_list.append(
                    LayerCompressionConfig(is_low_rank=True,
                                           bit_width_quantization_a=na,
                                           bit_width_quantization_b=nb,
                                           rank=r,
                                           per_channel_a=self.compression_config.per_channel_Ar,
                                           per_channel_b=self.compression_config.per_channel_Br))

    def _get_rank_options_for_bitwidths(self, n_Ar, n_Br, in_bit_widths, base_size, simd_fn):
        index = simd_fn(self.n_out) * self.rank_range_array * n_Ar + n_Br * self.rank_range_array * self.n_in < max(
            in_bit_widths) * base_size
        return self.rank_range_array[index]

    @staticmethod
    def _simd_fn(in_d, simd):
        return math.ceil(in_d / simd) * simd
