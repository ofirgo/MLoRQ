from copy import deepcopy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from compression.configs.compression_config import CompressionConfig, ABSPLIT, SVDScores, \
    ThresholdMethod, ParetoCost
from compression.configs.layer_comrpression_config import LayerCompressionConfig
from compression.module.compression_options import CompressionOptions
from compression.ordering import generate_point_ordering
from compression.quantization.adaround_utils import modified_soft_quantization
from compression.quantization.quantization import uniform_quantization, search_weights_scale_perc
from utils.memory_utils import get_gpu_memory_map

BYTE_SCALE = 8


class CompressionWrapper(nn.Module):
    def __init__(self, in_module, in_compression_config: CompressionConfig, node_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.add_module("base_module", in_module.to(in_module.weight.device))
        except Exception as e:
            get_gpu_memory_map()
            raise torch.OutOfMemoryError(e)

        try:
            self.add_module("pre_base_module", deepcopy(in_module))
        except Exception as e:
            get_gpu_memory_map()
            raise torch.OutOfMemoryError(e)
        self.pre_base_module.bias = None

        self.is_matmul = isinstance(in_module, nn.Linear)
        self.node_name = node_name
        self.is_conv = False
        grouped_conv = False
        if isinstance(in_module, nn.Conv2d):
            self.is_conv = True
            grouped_conv = in_module.groups > 1
            self.is_matmul = (in_module.kernel_size[0] == 1 and in_module.kernel_size[1] == 1)
        self.grouped_conv = grouped_conv

        self.register_parameter('weight', nn.Parameter(in_module.weight.clone().detach()))

        self.output_channels = self.weight.shape[0]
        self.max_bit_width = max(in_compression_config.weight_bit_list)

        r_weight = self.reshape_per_channel()
        self.maximal_rank = min(r_weight.shape[0], r_weight.shape[1])
        self.minimal_rank = self.weight.numel() / (r_weight.shape[0] + r_weight.shape[1])

        self.n_in = self.weight.shape[0]
        self.n_out = self.weight.shape[1]

        self.pareto_config = []
        self.pareto = []
        self.res_q = None
        self.res_lrq = None

        self.compression_active = False

        self.w_hessian = torch.ones_like(self.weight)
        self.f_score = None
        self.enable_low_rank = self.is_matmul

        self.A = self.B = None
        self.singular_values = None

        self.compression_options = None

        # AdaRound parameters
        self.reconstructed = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.qparam_train = False

        self.train_weight_scale = False
        self.inited = False
        self.channel_wise = True
        self.store_input = None
        self.bias = in_module.bias
        if self.is_conv:
            self.stride = in_module.stride
            self.padding = in_module.padding
            self.dilation = in_module.dilation
            self.groups = in_module.groups
        else:
            self.out_features = in_module.out_features

    def _init_weights_candidates_with_global_thresholds(self, in_weights, in_bit_widths, hessian):
        A_per_bit_delta = {}
        B_per_bit_delta = {}
        A_per_bit_zero_point = {}
        B_per_bit_zero_point = {}
        for nbits in in_bit_widths:
            d_A, z_A = search_weights_scale_perc(x=self.A, n_bits=nbits, hessian=hessian,
                                                 channel_wise=True,
                                                 x_ref=in_weights, x_complement=self.B)

            d_B, z_B = search_weights_scale_perc(x=self.B, n_bits=nbits, channel_wise=True)

            A_per_bit_delta[nbits] = d_A
            B_per_bit_delta[nbits] = d_B
            A_per_bit_zero_point[nbits] = z_A
            B_per_bit_zero_point[nbits] = z_B

        for co in self.compression_options.get_compression_options_list():
            if co.rank is None:
                delta, zero_point = search_weights_scale_perc(x=in_weights,
                                                              n_bits=co.bit_width_quantization,
                                                              channel_wise=True,
                                                              hessian=hessian)
                co.set_weights_params(delta, zero_point)
            else:
                co.set_weights_params_a(A_per_bit_delta[co.bit_width_quantization_a],
                                        A_per_bit_zero_point[co.bit_width_quantization_a])
                co.set_weights_params_b(B_per_bit_delta[co.bit_width_quantization_b],
                                        B_per_bit_zero_point[co.bit_width_quantization_b])

    def init_layer_reconstruction(self):
        self.register_buffer('original_weights', self.base_module.weight.clone().detach())

        if self.only_quantization:
            _alpha = self._init_alpha(self.original_weights.data.detach(),
                                      self.base_n_bits, self.delta.detach())
            self.register_parameter('alpha', nn.Parameter(_alpha))
        else:
            Ar = self.A[:, :self.selected_rank]
            Br = self.B[:self.selected_rank, :]
            t_rank_Ar = self.delta_a[:, :self.selected_rank]
            t_rank_Br = self.delta_b[:self.selected_rank, :]

            _alpha_a = self._init_alpha(Ar.data.detach(), self.n_Ar, t_rank_Ar.detach())
            self.register_parameter('alpha_a', nn.Parameter(_alpha_a))

            _alpha_b = self._init_alpha(Br.data.detach(), self.n_Br, t_rank_Br.detach())
            self.register_parameter('alpha_b', nn.Parameter(_alpha_b))
        self.reconstructed = True

    def _init_alpha(self, w: torch.Tensor, b: int, th: torch.Tensor, eps=1e-8):
        delta = th
        w_floor = torch.floor(w / (delta + eps))
        rest = (w / (delta + eps)) - w_floor  # rest of rounding [0, 1)
        alpha = -self._safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest
        return alpha

    def init_layer_compression(self, in_compression_config: CompressionConfig, output_ref,
                               representative_data_loader=None, qm=None, model_manager=None, hessian_mp=False,
                               config_to_set=None):

        if in_compression_config.threshold_method == ThresholdMethod.HMSE:
            hessian = self.w_hessian.to(self.get_weights_device())
        else:
            hessian = None

        if config_to_set is None:
            in_bit_widths = in_compression_config.weight_bit_list
            self.compression_options = CompressionOptions(self.n_out, self.n_in, in_compression_config,
                                                          self.enable_low_rank, self.weight.device)
            with torch.no_grad():
                in_weights = self.weight
                base_size = self.n_in * self.n_out

                if self.enable_low_rank:
                    w_scores = self._compute_weighted_svd_scores(in_compression_config.svd_scores)
                    u, s, v = self._compute_svd(in_compression_config, w_scores)

                    self.singular_values = s.flatten()

                    self._create_ab_split(u, v, s, in_compression_config.ab_split,
                                          w_scores)  # initializes A, B inplace

                    # Init candidates with a global threshold (for all ranks)
                    self._init_weights_candidates_with_global_thresholds(in_weights, in_bit_widths, hessian=hessian)
                else:
                    # Init candidates with per-candidate threshold
                    self._init_weights_quantization_for_candidates(in_weights, hessian=hessian)

                if in_compression_config.pareto_cost == ParetoCost.HMSEPerOutChannel or hessian_mp:
                    hessian = self.w_hessian.to(self.get_weights_device())
                else:
                    hessian = torch.ones_like(self.weight)
                pareto = generate_point_ordering(self.reshape_per_channel(),
                                                 self.compression_options,
                                                 base_size, self.n_in, self.n_out,
                                                 in_a=self.A, in_b=self.B,
                                                 in_cc=in_compression_config,
                                                 hessian_for_pareto=hessian)

            self.pareto_config = [p[2] for p in pareto]
            self.pareto = [p[:2] for p in pareto]
        else:
            # given config to set
            if len(config_to_set) == 1:
                bit_width = config_to_set[0]
                assert bit_width in in_compression_config.weight_bit_list
                delta, zero_point = search_weights_scale_perc(x=self.weight, n_bits=bit_width,
                                                              channel_wise=self.channel_wise,
                                                              hessian=hessian)
                co = LayerCompressionConfig(is_low_rank=False, bit_width_quantization=bit_width)
                co.set_weights_params(delta, zero_point)

            elif len(config_to_set) == 3:
                a_bits, b_bits, rank = config_to_set
                w_scores = self._compute_weighted_svd_scores(in_compression_config.svd_scores)
                u, s, v = self._compute_svd(in_compression_config, w_scores)

                self.singular_values = s.flatten()

                self._create_ab_split(u, v, s, in_compression_config.ab_split, w_scores)  # initializes A, B, C inplace

                d_A, z_A = search_weights_scale_perc(x=self.A[:, :rank], n_bits=a_bits,
                                                     hessian=hessian,
                                                     channel_wise=True,
                                                     x_ref=self.weight, x_complement=self.B[:rank, :])

                d_B, z_B = search_weights_scale_perc(x=self.B[:rank, :], n_bits=b_bits, channel_wise=True)

                co = LayerCompressionConfig(is_low_rank=True, rank=rank,
                                            bit_width_quantization_a=a_bits, bit_width_quantization_b=b_bits)
                co.set_weights_params_a(d_A, z_A)
                co.set_weights_params_b(d_B, z_B)
            else:
                raise Exception("Unexpected config to set")

            self.compression_options = CompressionOptions(self.n_out, self.n_in, in_compression_config,
                                                          self.enable_low_rank, self.weight.device,
                                                          compression_options=[co])

    def _init_weights_quantization_for_candidates(self, in_weights, hessian):
        for co in tqdm(self.compression_options.get_compression_options_list(),
                       "Running param search for compression candidates"):
            delta, zero_point = search_weights_scale_perc(x=in_weights, n_bits=co.bit_width_quantization,
                                                          channel_wise=self.channel_wise, hessian=hessian)
            co.set_weights_params(delta, zero_point)

    def set_compression_config(self, compression_config: LayerCompressionConfig):
        dtype = self.weight.dtype
        if compression_config.delta is None:
            if hasattr(self, 'delta'):
                del self.delta
        else:
            self.register_parameter('delta', nn.Parameter(compression_config.delta.type(dtype)))

        if compression_config.zero_point is None:
            if hasattr(self, 'zero_point'):
                del self.zero_point
        else:
            self.register_parameter('zero_point', nn.Parameter(compression_config.zero_point.type(dtype)))

        self.only_quantization = not compression_config.is_low_rank
        self.selected_rank = compression_config.rank

        if compression_config.delta_a is None:
            if hasattr(self, 'delta_a'):
                del self.delta_a
        else:
            self.register_parameter('delta_a', nn.Parameter(
                compression_config.delta_a.type(dtype)))

        if compression_config.zero_point_a is None:
            if hasattr(self, 'zero_point_a'):
                del self.zero_point_a
        else:
            self.register_parameter('zero_point_a', nn.Parameter(
                compression_config.zero_point_a.type(dtype)))

        if compression_config.delta_b is None:
            if hasattr(self, 'delta_b'):
                del self.delta_b
        else:
            self.register_parameter('delta_b', nn.Parameter(
                compression_config.delta_b.type(dtype)))

        if compression_config.zero_point_b is None:
            if hasattr(self, 'zero_point_b'):
                del self.zero_point_b
        else:
            self.register_parameter('zero_point_b', nn.Parameter(
                compression_config.zero_point_b.type(dtype)))

        self.base_n_bits = compression_config.bit_width_quantization
        self.n_Ar = compression_config.bit_width_quantization_a
        self.n_Br = compression_config.bit_width_quantization_b

    def compress_weight(self):
        if self.only_quantization:
            if self.reconstructed:
                gradient_factor = self._lsq_g_factor(self.original_weights,
                                                     self.base_n_bits) if self.train_weight_scale else 1.0
                w = modified_soft_quantization(self.original_weights, self.alpha, self.base_n_bits,
                                               self.delta, self.gamma, self.zeta, zero_point=self.zero_point,
                                               training=self.training,
                                               gradient_factor=gradient_factor)
            else:
                w = uniform_quantization(self.weight, self.delta, self.zero_point, 2 ** self.base_n_bits)

            return w.type(self.weight.dtype)

        else:
            Ar = self.A[:, :self.selected_rank]
            Br = self.B[:self.selected_rank, :]
            delta_rank_Ar = self.delta_a[:, :self.selected_rank]
            delta_rank_Br = self.delta_b[:self.selected_rank, :]
            zero_point_rank_Ar = self.zero_point_a[:, :self.selected_rank]
            zero_point_rank_Br = self.zero_point_b[:self.selected_rank, :]
            if self.reconstructed:
                gradient_factor = self._lsq_g_factor(Ar, self.n_Ar) if self.train_weight_scale else 1.0
                q_Ar = modified_soft_quantization(Ar, self.alpha_a, self.n_Ar,
                                                  delta_rank_Ar, self.gamma, self.zeta, zero_point_rank_Ar,
                                                  training=self.training,
                                                  gradient_factor=gradient_factor)
                gradient_factor = self._lsq_g_factor(Br, self.n_Br) if self.train_weight_scale else 1.0
                q_Br = modified_soft_quantization(Br, self.alpha_b, self.n_Br,
                                                  delta_rank_Br, self.gamma, self.zeta, zero_point_rank_Br,
                                                  training=self.training,
                                                  gradient_factor=gradient_factor)
            else:
                q_Ar = uniform_quantization(Ar, delta_rank_Ar, zero_point_rank_Ar, 2 ** self.n_Ar)
                q_Br = uniform_quantization(Br, delta_rank_Br, zero_point_rank_Br, 2 ** self.n_Br)

            if self.is_conv:
                q_Ar = q_Ar.unsqueeze(dim=-1).unsqueeze(dim=-1)
                q_Br = q_Br.unsqueeze(dim=-1).unsqueeze(dim=-1)
            return q_Ar.type(self.weight.dtype), q_Br.type(self.weight.dtype)

    def get_trainable_params(self):
        params = []
        if self.reconstructed:
            if self.only_quantization:
                params.append(self.alpha)
            else:
                params.extend([self.alpha_a, self.alpha_b])
        else:
            if self.only_quantization:
                params.append(self.weight)
            else:
                params.extend([self.A, self.B])
        bias_params = [self.base_module.bias] if self.base_module.bias is not None else None
        if self.only_quantization:
            scale_params = [self.delta]
        else:
            scale_params = [self.delta_a, self.delta_b]

        return params, bias_params, scale_params

    def get_weights_device(self):
        return self.weight.device

    def reshape_per_channel(self):
        return self.weight.reshape([self.output_channels, -1])

    def _compute_svd(self, in_compression_config, w_scores):
        if in_compression_config.svd_scores == SVDScores.ID:
            U, S, Vh = torch.linalg.svd(self.reshape_per_channel().type(torch.float32), full_matrices=False)

            w_dtype = self.weight.dtype
            return (U.type(w_dtype),
                    S.type(w_dtype),
                    Vh.type(w_dtype))

        return self._weighted_svd(w_scores)

    @staticmethod
    def _safe_log(x, eps):
        return torch.log(torch.max(x, torch.Tensor([eps]).to(x.device)))

    def set_train_weight_scale(self):
        self.train_weight_scale = True

    def reset_layer_reconstruction(self):
        if hasattr(self, 'alpha'):
            del self.alpha
        if hasattr(self, 'alpha_a'):
            del self.alpha_a
        if hasattr(self, 'alpha_b'):
            del self.alpha_b
        self.weight.data = self.original_weights.data
        if hasattr(self, 'A') and self.A is not None:
            self.A.data = self.original_A.data
        if hasattr(self, 'B') and self.B is not None:
            self.B.data = self.original_B.data
        self.reconstructed = False

    @property
    def hessian_per_channel(self):
        return self.w_hessian.to(self.get_weights_device()).reshape(
            [self.output_channels, -1]).sum(dim=-1)

    def _compute_weighted_svd_scores(self, svd_scores_method: SVDScores):
        w_scores = None
        if svd_scores_method == SVDScores.LFH:
            w_scores = self.w_hessian.to(self.get_weights_device()).sum(dim=-1)
            w_scores = torch.sqrt(w_scores) + 1e-8
            w_scores = torch.diag(w_scores)

        return w_scores

    def _create_ab_split(self, u, v, s, ab_split: ABSPLIT, w_scores: torch.Tensor):
        if ab_split == ABSPLIT.US_V:
            if w_scores is None:
                self.A = nn.Parameter(u @ torch.diag(s))  # Fold s into u.
            else:
                self.A = nn.Parameter(torch.linalg.inv(w_scores).type(u.dtype) @ u @ torch.diag(s))  # Fold s into u.
            self.B = nn.Parameter(v)
        else:
            if w_scores is None:
                self.A = nn.Parameter(u)
            else:
                self.A = nn.Parameter(torch.linalg.inv(w_scores).type(u.dtype) @ u)

            self.B = nn.Parameter(torch.diag(s) @ v)  # Fold s into v.

        self.register_buffer('original_A', self.A.data)
        self.register_buffer('original_B', self.B.data)

    def _weighted_svd(self, w_scores):
        Hw = w_scores @ self.reshape_per_channel()

        u, s, v = torch.linalg.svd(Hw,
                                   full_matrices=False)

        return u, s, v

    def add_weights_hessian_information(self, w_hessian):
        self.w_hessian = w_hessian

    @staticmethod
    def _lsq_g_factor(w, b):
        return 1 / float(torch.sqrt(torch.numel(w) * (2 ** (b - 1) - 1)))

    def get_layer_compression_config(self):
        return self.base_n_bits, self.selected_rank / self.maximal_rank

    def enable_compression(self):
        self.compression_active = True

    def disable_compression(self):
        self.compression_active = False

    @property
    def n_compression_options(self):
        return len(self.pareto_config)

    def compute_float_size(self):
        return self.weight.nbytes

    def compute_size(self, cfg):
        if not cfg.is_low_rank:
            return cfg.bit_width_quantization * self.weight.numel() / BYTE_SCALE
        else:
            assert cfg.bit_width_quantization_a is not None and cfg.bit_width_quantization_b is not None and cfg.rank is not None
            return int((self.A.shape[0] * cfg.bit_width_quantization_a + self.B.shape[
                1] * cfg.bit_width_quantization_b) * cfg.rank / BYTE_SCALE)

    def _get_bops_input_scale(self):
        if isinstance(self.base_module, nn.Linear):
            if len(self.input_shape) > 1:
                scale = np.prod(self.input_shape[:-1])
            else:
                scale = 1
        elif isinstance(self.base_module, nn.Conv2d):
            if len(self.input_shape) == 3:  # Assume batch is remove
                scale = ((self.input_shape[1] + self.padding[0]) / self.stride[0]) * (
                        (self.input_shape[2] + self.padding[1]) / self.stride[1])
            else:
                raise Exception()
        else:
            raise Exception("Unexpected op for bops computation")

        return scale

    def compute_float_bops(self):
        scale = self._get_bops_input_scale()
        float_bits = 8 * self.weight.dtype.itemsize
        return scale * float_bits * self.weight.numel() * float_bits

    def compute_bops(self, cfg, act_nbits):
        scale = self._get_bops_input_scale()

        if not cfg.is_low_rank:
            assert cfg.bit_width_quantization is not None
            return scale * cfg.bit_width_quantization * self.weight.numel() * act_nbits
        else:
            assert cfg.bit_width_quantization_a is not None and cfg.bit_width_quantization_b is not None and cfg.rank is not None
            cost_a = scale * cfg.bit_width_quantization_a * self.A.shape[0] * cfg.rank * 16
            cost_b = scale * cfg.bit_width_quantization_b * self.B.shape[1] * cfg.rank * act_nbits

        return (cost_a + cost_b).item()


    def compute_bops_flat(self, act_nbits, is_low_rank, bit_width_quantization=None, bit_width_quantization_a=None,
                          bit_width_quantization_b=None, rank=None):
        scale = self._get_bops_input_scale()

        if not is_low_rank:
            assert bit_width_quantization is not None
            return scale * bit_width_quantization * self.weight.numel() * act_nbits
        else:
            assert bit_width_quantization_a is not None and bit_width_quantization_b is not None and rank is not None
            cost_a = scale * bit_width_quantization_a * self.n_out * rank * 16
            cost_b = scale * bit_width_quantization_b * self.n_in * rank * act_nbits

        return (cost_a + cost_b).item()


    def forward(self, x):
        if self.compression_active:
            if self.only_quantization:
                w = self.compress_weight()
                if self.is_conv:
                    return torch.nn.functional.conv2d(x, w, self.base_module.bias, self.base_module.stride,
                                                      self.base_module.padding, self.base_module.dilation,
                                                      self.base_module.groups)
                else:
                    return torch.nn.functional.linear(x, w, self.base_module.bias)
            else:
                q_Ar, q_Br = self.compress_weight()

                if self.is_conv:
                    z = torch.nn.functional.conv2d(x, q_Br, self.base_module.bias, self.pre_base_module.stride,
                                                   self.pre_base_module.padding, self.pre_base_module.dilation,
                                                   self.pre_base_module.groups)
                else:
                    z = torch.nn.functional.linear(x, q_Br)

                if self.is_conv:
                    return torch.nn.functional.conv2d(z, q_Ar, self.base_module.bias, self.pre_base_module.stride,
                                                      self.pre_base_module.padding, self.pre_base_module.dilation,
                                                      self.pre_base_module.groups)
                else:
                    return torch.nn.functional.linear(z, q_Ar, self.base_module.bias)
        else:
            self.base_module.weight.data = self.weight.detach()
            return self.base_module(x)
