from dataclasses import dataclass

import torch

from compression.quantization.quantization import uniform_quantization


@dataclass
class LayerCompressionConfig:
    is_low_rank: bool
    bit_width_quantization: int = None
    threshold_quantization: torch.Tensor = None
    delta: torch.Tensor = None
    zero_point: torch.Tensor = None
    bit_width_quantization_a: int = None
    bit_width_quantization_b: int = None
    threshold_quantization_a: torch.Tensor = None
    threshold_quantization_b: torch.Tensor = None
    delta_a: torch.Tensor = None
    delta_b: torch.Tensor = None
    zero_point_a: torch.Tensor = None
    zero_point_b: torch.Tensor = None
    rank: int = None
    per_channel_a: bool = True
    per_channel_b: bool = True
    scale: torch.Tensor = None
    scale_inverse: torch.Tensor = None

    def set_weights_params(self, delta, zero_point):
        self.delta, self.zero_point = delta, zero_point

    def set_weights_params_a(self, delta, zero_point):
        self.delta_a, self.zero_point_a = delta, zero_point

    def set_weights_params_b(self, delta, zero_point):
        self.delta_b, self.zero_point_b = delta, zero_point

    def quantize_Ar_uniform(self, Ar, rank):
        delta = self.delta_a[:, :rank]
        zero_point = self.zero_point_a[:, :rank]
        return uniform_quantization(Ar, delta, zero_point, 2 ** self.bit_width_quantization_a)

    def quantize_Br_uniform(self, Br, rank):
        delta = self.delta_b[:rank, :]
        zero_point = self.zero_point_b[:rank, :]
        return uniform_quantization(Br, delta, zero_point, 2 ** self.bit_width_quantization_b)

    def size(self, n_in, n_out):
        if self.rank is None:
            return n_in * n_out * self.bit_width_quantization

        return self.rank * n_out * self.bit_width_quantization_a + self.rank * n_in * self.bit_width_quantization_b
