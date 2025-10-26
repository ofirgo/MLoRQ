import operator

import torch

from utils.device_management import available_devices

AVAILABLE_DEVICES = available_devices()
DEVICE = AVAILABLE_DEVICES[0]  # Keep original expected single device semantics
PRIMARY_DEVICE = DEVICE  # Alias for clarity where code expects primary device

SOLVER_TIME_LIMIT = 120  # 2 minutes
FLOAT_BIT_WIDTH = 32
MP_BIT_WIDTH = [2, 3, 4, 6, 8]
LINEAR_OPS = [(torch.nn.Conv1d,),
              (torch.nn.Conv2d,),
              (torch.nn.Conv3d,),
              (torch.nn.Linear,)]

ACTIVATION_OPS = [(torch.nn.ReLU,),
                  (torch.nn.ReLU6,),
                  (torch.nn.Identity,)]

LINEAR_QUANTIZE_OPERATORS = [torch.nn.Linear, torch.matmul, operator.matmul]
REDUNDANT_OPERATORS = [torch.nn.Identity, torch.nn.Dropout]

SIGMOID_MINUS = 4

PARAM_SEARCH_ITERS = 15
PARAM_SEARCH_STEPS = 100

ORIGINAL_W = 'original_w'
SVD_W_SCORES = 'svd_w_scores'

CALL_MODULE = 'call_module'
CALL_METHOD = 'call_method'
CALL_FUNCTION = 'call_function'
ACTIVATION_QUANT_STR = 'activation_quant'

TORCHVISION = 'torchvision'
TIMM = 'timm'
