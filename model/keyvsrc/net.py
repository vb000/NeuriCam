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
from basicvsr_pp.basicvsr_pp import BasicVSRPlusPlus
from basicvsr_pp.basicvsr_net import ResidualBlocksWithInputConv

class Net(nn.Module):
    def __init__(self, upscale_factor, mid_channels, upsampler='default'):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsampler = upsampler

        self.feat_extract_lr = ResidualBlocksWithInputConv(1, mid_channels, 5)
        self.feat_extract_key = nn.Sequential(
            nn.Conv2d(3, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))
        self.basicvsr_pp = BasicVSRPlusPlus(
            mid_channels=mid_channels,
            num_blocks=7,
            max_residue_magnitude=10,
            is_low_res_input=True,
            spynet_pretrained=os.path.join(os.path.dirname(__file__),
                                           'spynet_20210409-c6c1bd09.pth'),
            cpu_cache_length=100)
        self.basicvsr_pp.feat_extract = None
        if upsampler == 'attn':
            self.basicvsr_pp.reconstruction = None
            self.reconstruction = ResidualBlocksWithInputConv(mid_channels, mid_channels, 5)

    def extract_features(self, lr, key, key_frame_int):
        def _feature_extract(seq, extractor, scale):
            n, t, c, h, w = seq.size()
            _feats = extractor(seq.view(-1, c, h, w))
            _feats = _feats.view(n, t, -1, h//scale, w//scale)
            return _feats

        feats_lr = _feature_extract(lr, self.feat_extract_lr, 1)
        feats_key = _feature_extract(key, self.feat_extract_key, self.upscale_factor)

        feats = {'spatial': []}
        for i in range(lr.size(1)):
            if i % key_frame_int == 0:
                feats['spatial'].append(feats_key[:, int(i/key_frame_int), :, :, :])
            feats['spatial'].append(feats_lr[:, i, :, :, :])
        return feats

    def propagate(self, lr, feats, key_frame_int):
        n, t, c, h, w = lr.size()

        # Exapnd lr sequence with an additional frame
        # for every key-frame.
        lr_e = []
        key_indices = []
        for i in range(t):
            if i % key_frame_int == 0:
                lr_e.append(lr[:, i, :, :, :])
                key_indices.append(len(lr_e) - 1)
            lr_e.append(lr[:, i, :, :, :])
        lr_e = torch.stack(lr_e, dim=1)

        # whether to cache the features in CPU
        self.basicvsr_pp.cpu_cache = False

        # compute optical flow using the low-res inputs
        assert h >= 64 and w >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.basicvsr_pp.compute_flow(lr_e)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.basicvsr_pp.propagate(feats, flows, module)

        # Remove output features corresponding to key-frames
        feats_e = {}
        for k in feats.keys():
            feats_e[k] = []
            for i, f in enumerate(feats[k]):
                if i not in key_indices:
                    feats_e[k].append(f)

        return feats_e

    def attn_similarity_upsampler(self, feats):
        outputs = []
        for t in range(len(feats['spatial'])):
            # Aggregate features
            cummulative_feats = [feats[k][t] for k in feats if k != 'spatial']
            similarities = [torch.sum(cf*feats['spatial'][t], dim=1)
                            for cf in cummulative_feats]
            similarities = torch.stack(similarities, dim=1)
            weights = torch.softmax(similarities, dim=1)
            aggr_feat = torch.zeros_like(cummulative_feats[0])
            for i in range(len(cummulative_feats)):
                aggr_feat += cummulative_feats[i] * weights[:, i:i+1, :, :]

            # Upsample
            hr = self.reconstruction(aggr_feat)
            hr = self.basicvsr_pp.lrelu(self.basicvsr_pp.upsample1(hr))
            hr = self.basicvsr_pp.lrelu(self.basicvsr_pp.upsample2(hr))
            hr = self.basicvsr_pp.lrelu(self.basicvsr_pp.conv_hr(hr))
            hr = torch.tanh(self.basicvsr_pp.conv_last(hr))

            outputs.append(hr)
        return torch.stack(outputs, dim=1)

    def forward(self, batch):
        lr = batch['LR']
        key = batch['key']
        key_frame_int = batch['key_frame_int'][0]

        # Normalize to [-1, 1] range
        lr = 2. * lr - 1.
        key = 2. * key - 1.

        outputs = []
        for s in range(key.size(1)):
            start_index = s * key_frame_int
            if s < key.size(1) - 1:
                lr_s = lr[:, start_index:start_index+key_frame_int+1, :, :, :].clone()
                key_s = key[:, s:s+2, :, :, :].clone()
            else:
                lr_s = lr[:, start_index:, :, :, :].clone()
                key_s = key[:, s:s+1, :, :, :].clone()

            feats = self.extract_features(lr_s, key_s, key_frame_int)
            feats = self.propagate(lr_s, feats, key_frame_int)
            if self.upsampler == 'attn':
                out_s = self.attn_similarity_upsampler(feats)
            else:
                out_s = self.basicvsr_pp.upsample(lr_s, feats)

            outputs += [key[:, s, :, :, :]]
            if s < key.size(1) - 1:
                outputs += torch.unbind(out_s[:, 1:-1, :, :, :], dim=1)
            else:
                outputs += torch.unbind(out_s[:, 1:, :, :, :], dim=1)

        outputs = torch.stack(outputs, dim=1)
        outputs = (outputs + 1.) / 2.
        return {'HR_lab': outputs}

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

def get_item(GT, LR_, key, upscale_factor, key_frame_int, downsampling_method=None, noise_fn=None):
    GT_lab = utils.tensor_rgb2lab(GT)

    # Load grayscale LR frame convert it to single channel frame
    LR = utils.tensor_rgb2lab(LR_)
    LR = LR[:, 0:1, :, :]
    
    key = utils.tensor_rgb2lab(key)

    train_item = {'LR': LR, 'key': key, 'key_frame_int': key_frame_int}
    target = {'HR_lab': GT_lab, 'HR': GT, 'key_frame_int': key_frame_int}
    return train_item, target
