import os
os.environ['TIMM_FUSED_ATTN'] = '0'

import timm
import torch
import torchvision
import yaml
from torch import fx
from torch.utils.data import Subset

from constants import TIMM, TORCHVISION, PRIMARY_DEVICE
from helpers.classification_evalution import vision_classification_evaluation
from model_managers.base_model import BaseModel, ModelManagerArgs


class VisionModel(BaseModel):

    def __init__(self, model_args: ModelManagerArgs):
        super().__init__(model_args)
        self.data_config = timm.data.resolve_model_data_config(self.float_model)

        self.train_dir = model_args.train_dir
        self.val_dir = model_args.val_dir

        # init datasets
        self.calib_ds = timm.data.create_dataset("", self.train_dir,
                                                 transform=timm.data.create_transform(**self.data_config,
                                                                                      is_training=False))
        self.training_ds = timm.data.create_dataset("", self.train_dir,
                                                    transform=timm.data.create_transform(**self.data_config,
                                                                                         is_training=True))
        self.init_random_samples_indices(model_args.num_samples)

    def _init_float_model(self):
        with open('models_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)["vision_models"]
        self.model_config = model_config[self.model_name]
        self.float_accuracy = self.model_config["float_eval_results"]
        model_lib = self.model_config.get('model_lib', TIMM)
        if model_lib == TORCHVISION:
            float_model = getattr(torchvision.models,
                                  self.model_config["orig_model_name"])(weights=
                                                                        self.model_config["weights"]).to(self.device)
        elif model_lib == TIMM:
            float_model = timm.create_model(self.model_config["orig_model_name"], pretrained=True).to(self.device)
        else:
            raise NotImplementedError
        return float_model

    def get_fx_graph(self, model):
        return fx.symbolic_trace(model)

    def evaluate(self, model, eval_data_loader):
        return vision_classification_evaluation(model, eval_data_loader)

    def forward(self, model, data):
        return model(data)

    def data_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            data, labels = batch[0], batch[1]
        else:
            data = batch
        return data.to(self.device)

    def init_random_samples_indices(self, num_samples):
        self.samples_indices = torch.randperm(len(self.calib_ds))[:num_samples].tolist()

    def get_consistent_sub_dataset(self, is_training):
        if is_training:
            return Subset(self.training_ds, self.samples_indices)
        else:
            return Subset(self.calib_ds, self.samples_indices)

    def get_representative_dataset(self, sub_ds, shuffle, is_training):
        sub_ds = self.get_consistent_sub_dataset(is_training)
        representative_data_loader = torch.utils.data.DataLoader(sub_ds,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=shuffle,
                                                                 num_workers=4 if PRIMARY_DEVICE.type == 'cuda' else 0,
                                                                 pin_memory=(PRIMARY_DEVICE.type == 'cuda'),
                                                                 )
        return representative_data_loader

    def get_validation_data_loader(self):
        val_dataset = timm.data.create_dataset(name='Imagenet', root=self.val_dir, is_training=False,
                                               batch_size=self.batch_size)

        pre_process_dict = {'input_size': self.data_config['input_size'],
                            'interpolation': self.data_config['interpolation'],
                            'mean': self.data_config['mean'],
                            'std': self.data_config['std'],
                            'crop_pct': self.data_config['crop_pct']}

        if self.model_config.get('pre_process') is not None:
            if self.model_config['pre_process'].get('input_size') is not None:
                pre_process_dict['input_size'] = self.model_config['pre_process'].get('input_size')

            if self.model_config['pre_process'].get('interpolation') is not None:
                pre_process_dict['interpolation'] = self.model_config['pre_process'].get('interpolation')

            if self.model_config['pre_process'].get('mean') is not None:
                pre_process_dict['mean'] = self.model_config['pre_process'].get('mean')

            if self.model_config['pre_process'].get('std') is not None:
                pre_process_dict['std'] = self.model_config['pre_process'].get('std')

            if self.model_config['pre_process'].get('crop_pct') is not None:
                pre_process_dict['crop_pct'] = self.model_config['pre_process'].get('crop_pct')

        num_workers = 10 if PRIMARY_DEVICE.type == 'cuda' else 0
        val_data_loader = timm.data.create_loader(
            val_dataset,
            input_size=pre_process_dict['input_size'],
            batch_size=self.batch_size,
            interpolation=pre_process_dict['interpolation'],
            mean=pre_process_dict['mean'],
            std=pre_process_dict['std'],
            crop_pct=pre_process_dict['crop_pct'],
            num_workers=num_workers,
            use_prefetcher=(PRIMARY_DEVICE.type == 'cuda'),
            persistent_workers=(num_workers > 0))
        return val_data_loader
