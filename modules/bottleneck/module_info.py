from math import inf
import torch
from tqdm import tqdm
from scipy.stats import norm
import numpy as np
import torch.nn as nn
from utils.misc import to_np


class ModuleInfo():
    """
        Save the informations from a module (to avoid having to collect them everytime)
    """
    def __init__(self, module):
        self.type = type(module)
        self.module = module 
        self.conv_info = {'h':0, 'w':0, 'k':0, 'cin':0, 'cout':0, 'has_bias': False}

    def conv_info(self):
        return self.conv_info

    def module(self):
        return self.module


class ModulesInfo:
    """
    A wrapper for a list of modules.
    Allows to collect data for all modules in one feed-forward.
    Args:
        - model: the model to collect data from
        - modulesInfo: a list of ModuleInfo objects
        - input_img_size: the size of the input image, for the feed-forwarding
        - device: the device of the model
    """
    def __init__(self, model, modulesInfo, input_img_size=None, device=None):
        self.model = model
        self.device = device
        self.modulesInfo = modulesInfo
        if input_img_size:
            input_image = torch.randn(1, 3, input_img_size, input_img_size).cuda()
            self.feed(input_image)

    def _make_feed_hook(self, i):
        def hook(m, x, z):
            self.modulesInfo[i].conv_info['cin'] = int(x[0].size(1))
            self.modulesInfo[i].conv_info['cout'] = int(z.size(1))
            self.modulesInfo[i].conv_info['has_bias'] = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Conv2d):
                self.modulesInfo[i].conv_info['h'] = int(z.size(2)) if len(z.size())>2 else 1
                self.modulesInfo[i].conv_info['w'] = int(z.size(3)) if len(z.size())>2 else 1
                self.modulesInfo[i].conv_info['k'] = int(m.weight.size(2))
            elif isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                self.modulesInfo[i].conv_info['h'] = int(z.size(2)) if len(z.size())>2 else 1
                self.modulesInfo[i].conv_info['w'] = int(z.size(3)) if len(z.size())>2 else 1
                self.modulesInfo[i].conv_info['k'] = int(x[0].size(2))//int(z.size(2))
            else:
                self.modulesInfo[i].conv_info['h'] = int(x[0].size(2)) if len(x[0].size())>2 else 1
                self.modulesInfo[i].conv_info['w'] = int(x[0].size(3)) if len(x[0].size())>2 else 1
        return hook

    def feed(self, input_image):
        hook_handles = [e.module.register_forward_hook(self._make_feed_hook(i)) for i, e in enumerate(self.modulesInfo)]

        if self.device is not None:
            self.model(input_image.to(self.device))
        else:
            self.model(input_image)

        for handle in hook_handles:
            handle.remove()

    def get_conv_info(self, i):
            return self.modulesInfo[i].info
    
    def modules(self):
        return [m for m in self.modulesInfo]