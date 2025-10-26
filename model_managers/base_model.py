import abc
import torch
from dataclasses import dataclass

from utils.device_management import available_devices
from constants import DEVICE


@dataclass
class ModelManagerArgs:
    model_name: str
    batch_size: int
    num_samples: int
    train_dir: str
    val_dir: str
    seed: int = 42
    train: bool = None


class BaseModel(abc.ABC):
    def __init__(self, model_args: ModelManagerArgs):
        self.model_name = model_args.model_name
        self.batch_size = model_args.batch_size
        self.num_samples = model_args.num_samples
        self.train = model_args.train
        self.seed = model_args.seed
        self.should_fuse_model = True
        self.device = DEVICE
        self.float_model = self._init_float_model()
        self.skip_quantization_layer_names = []

    @abc.abstractmethod
    def _init_float_model(self):
        pass

    @abc.abstractmethod
    def get_fx_graph(self, model):
        pass

    @abc.abstractmethod
    def get_representative_dataset(self, num_samples, shuffle, is_training):
        pass

    @abc.abstractmethod
    def get_validation_data_loader(self):
        pass

    @abc.abstractmethod
    def evaluate(self, model, eval_data_loader):
        pass

    @abc.abstractmethod
    def forward(self, model, data):
        pass

    @abc.abstractmethod
    def data_to_device(self, batch):
        pass

    def set_float_accuracy(self, acc):
        self.float_accuracy = acc
