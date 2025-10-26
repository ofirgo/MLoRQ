from torch.fx.experimental.optimization import fuse

from compression.module.compression_wrapper import CompressionWrapper
from compression.module.module_replacer import replace2quantized_model
from constants import LINEAR_OPS
from model_managers.base_model import BaseModel


def prepare_module(float_model, model_manager: BaseModel, in_cc):
    float_model = float_model.eval()
    if model_manager.should_fuse_model:
        float_model = fuse(float_model)
    float_model.to(model_manager.device)

    def replace_function(in_module, node_name):
        return CompressionWrapper(in_module, in_cc, node_name=node_name)

    compressed_model, float_model = replace2quantized_model(float_model, model_manager, replace_function,
                                                            linear_patterns=LINEAR_OPS), float_model

    return compressed_model, float_model
