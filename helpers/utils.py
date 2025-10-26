from compression.module.compression_wrapper import CompressionWrapper
from compression.quantization.activation_quantization import ActivationQuantizer


def is_compressed_layer(module, name=None):
    return module is not None and isinstance(module, CompressionWrapper)


def is_quantized_activation(module):
    return module is not None and isinstance(module, ActivationQuantizer)
