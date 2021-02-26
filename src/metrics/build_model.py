# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/build_model.py


from metrics.inception_network import InceptionV3
from utils.misc import *

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP



class build_eval_model(object):
    def __init__(self, model_name, distributed_data_parallel, world_size, local_rank, save_output):
        super(build_eval_model, self).__init__()
        self.model_name = model_name
        self.save_output = save_output
        if self.model_name == 'Inception_V3':
            self.model = InceptionV3().to(local_rank)
        elif self.model_name == 'SwAV':
            self.model = torch.hub.load('facebookresearch/swav', 'resnet50').to(local_rank)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(save_output)
                    hook_handles.append(handle)
        else:
            raise NotImplementedError

        if world_size > 1 and distributed_data_parallel:
            toggle_grad(self.model, on=True)
            self.model = DDP(self.model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
        elif world_size > 1 and distributed_data_parallel is False:
            self.model = DataParallel(self.model, output_device=local_rank)
        else:
            pass


    def eval(self):
        self.model.eval()


    def get_outputs(self, x):
        if self.model_name == 'Inception_V3':
            repres, logits = self.model(x)
        else:
            logits = self.model(x)
            repres = self.save_output.outputs[0][0]
            self.save_output.clear()
        return repres, logits

