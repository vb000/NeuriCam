import sys, os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils
from gaussian_downsample import gaussian_downsample
from basicvsr_pp.basicvsr_pp import BasicVSRPlusPlus

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.generator = BasicVSRPlusPlus(
            mid_channels=64,
            num_blocks=7,
            max_residue_magnitude=10,
            is_low_res_input=True,
            spynet_pretrained=os.path.join(os.path.dirname(__file__),
                                           'spynet_20210409-c6c1bd09.pth'),
            cpu_cache_length=100)

    def forward(self, batch):
        out = self.generator(batch['LR'])
        return {'HR': out}

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            # Edit
            # logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=None)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


def get_item(GT, LR, upscale_factor, downsampling_method=None, noise_fn=None):
    if LR is None:
        if downsampling_method == "blur_cv2": 
            T, C, H, W = GT.shape
     
            LR = []
            for frm in GT:
                frm = frm.permute(1, 2, 0).numpy() # [H, W, C]
                frm = cv2.GaussianBlur(frm, (13, 13), 1.6, cv2.BORDER_DEFAULT)
                frm = cv2.resize(frm, (int(W / 4.), int(H / 4.)), interpolation=cv2.INTER_CUBIC) # [H/4, W/4, C]
                frm = torch.from_numpy(frm).permute(2, 0, 1) # [C, H/4, W/4]
                LR.append(frm)
            LR = torch.stack(LR, dim=0) # [T, C, H/4, W/4]
        elif downsampling_method == "blur_script":
            GT = GT.permute(1, 0, 2, 3) # [C, T, H, W]
            LR = gaussian_downsample(GT, upscale_factor)
            LR = LR.permute(1,0,2,3)
            GT = GT.permute(1,0,2,3) # [T,C,H,W]

    if noise_fn is not None:
        LR = noise_fn(LR)

    train_item = {'LR': LR}
    target = {'HR': GT}
    return train_item, target
