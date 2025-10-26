from dataclasses import dataclass
from typing import List
from enum import Enum


class DecompositionMode(Enum):
    SINGLE_OP = "SINGLE_OP"
    SPLIT_OP = "SPLIT_OP"


class ABSPLIT(Enum):
    US_V = "US_V"
    U_SV = "U_SV"


class ThresholdMethod(Enum):
    MSE = 'MSE'
    HMSE = 'HMSE'


class SVDScores(Enum):
    ID = 'ID'
    LFH = 'LFH'


class ParetoCost(Enum):
    MSE = 'MSE'
    HMSEPerOutChannel = 'HMSEPerOutChannel'


@dataclass
class CompressionConfig:
    weight_bit_list: List
    ab_split: ABSPLIT
    per_channel_Ar: bool
    per_channel_Br: bool
    pareto_cost: ParetoCost
    svd_scores: SVDScores
    threshold_method: ThresholdMethod
    optimize_scale: bool = True
    simd: int = 1
    num_inter_points: int = 2
    activation_n_bits: int = 8
    activation_mp: bool = False
    disable_softmax_log_scale: bool = False
    disable_ln_reparam: bool = False
