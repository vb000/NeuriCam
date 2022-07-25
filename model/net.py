import sys, os
import logging
import math
import numpy as np
from itertools import chain
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append(os.path.dirname(__file__))

import utils
import keyvsrc.net as keyvsrc_module
from keyvsrc.net import Net as keyvsrc

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        if params.net == "keyvsrc":
            self.model = keyvsrc(params.upscale_factor, params.mid_channels, params.upsampler)
        else:
            sys.exit("Net %s doesn't exist" % params.net)

    def forward(self, model_in):
        return self.model(model_in)

class Metrics():
    def __init__(self, params, verbose=False):
        self.net = params.net
        self.border = 2 * params.upscale_factor if params.reflect_padding == 1 else 0
        self.metrics = ['PSNR_Y', 'PSNR_RGB', 'SSIM_Y', 'SSIM_RGB']
        self.verbose = verbose

    def __call__(self, pred, target):
        skip_frames = []
        grayscale = False

        if self.net == "keyvsrc":
            _pred = utils.tensor_lab2rgb(pred['HR_lab'])
            _target = target['HR']
        else:
            sys.exit("Net %s doesn't exist" % params.net)

        if self.net == "keyvsrc":
            num_frames = target['HR'].shape[1]
            key_frame_int = target['key_frame_int'][0]
            skip_frames = list(range(0, num_frames, key_frame_int))

        if self.border != 0:
            b = self.border
            _pred = _pred[:, :, :, b:-b, b:-b]
            _target = _target[:, :, :, b:-b, b:-b]

        return self._compute_metrics(_pred, _target, skip_frames, grayscale)

    def _compute_metrics(self, pred, target, skip_frames=[], grayscale=False):
        assert pred.shape == target.shape, "Prediction target shape mismatch"

        avg_metrics = {m: [] for m in self.metrics}

        # Clip values and crop borders
        batch1 = pred.numpy().clip(0., 1.)
        batch2 = target.numpy().clip(0., 1.)

        for i, (vid1, vid2) in enumerate(zip(batch1, batch2)):
            values = {m: [] for m in self.metrics}
            for j, (frm1, frm2) in enumerate(zip(vid1, vid2)):
                if j in skip_frames:
                    continue

                frm1 = (255. * frm1).round().astype(np.uint8)
                frm2 = (255. * frm2).round().astype(np.uint8)
                frm1 = frm1.transpose((1, 2, 0))[:, :, ::-1] # [H, W, BGR]
                frm2 = frm2.transpose((1, 2, 0))[:, :, ::-1]

                if grayscale:
                    values['PSNR_Y'].append(utils.psnr(frm1, frm2))
                    values['PSNR_RGB'].append(utils.psnr(frm1, frm2))
                    values['SSIM_Y'].append(utils.ssim(frm1, frm2))
                    values['SSIM_RGB'].append(utils.ssim(frm1, frm2))
                else:
                    frm1_y = 255. * utils.bgr2ycbcr(frm1 / 255., only_y=True)
                    frm2_y = 255. * utils.bgr2ycbcr(frm2 / 255., only_y=True)

                    values['PSNR_Y'].append(utils.psnr(frm1_y, frm2_y))
                    values['PSNR_RGB'].append(utils.psnr(frm1, frm2))
                    values['SSIM_Y'].append(utils.ssim(frm1_y, frm2_y))
                    values['SSIM_RGB'].append(utils.ssim(frm1, frm2))

                if self.verbose:
                    print('frame %d: PSNR_Y=%.04f PSNR_RGB=%.04f' %
                          (j, values['PSNR_Y'][-1], values['PSNR_RGB'][-1]))
            for m in self.metrics:
                avg_metrics[m].append(np.mean(values[m]))

        return avg_metrics

class CharbonnierLoss(nn.Module):
    """Charbonnierloss."""
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(torch.square(diff) + self.eps**2)
        loss = torch.mean(error) 
        return loss 

def configure_parameters(model, model_dir, params):
    if params.net == "keyvsrc":
        # Param groups
        spynet_params = []
        main_params = []
        for name, param in model.named_parameters():
            spynet_instance = 'model.basicvsr_pp.spynet'
            if spynet_instance == name[:len(spynet_instance)]:
                spynet_params.append(param)
            else:
                main_params.append(param)

        return [{'params': spynet_params, 'lr': params.spynet_lr},
                {'params': main_params, 'lr': params.main_lr}]

    return filter(lambda p: p.requires_grad, model.parameters())

# Loss function
def loss_fn(params):
    def l1_loss(X, Y):
        lfn = nn.L1Loss()
        loss = lfn(X['HR'], Y['HR'])
        return loss

    def sd_loss(X, Y):
        lfn = CharbonnierLoss()
        loss = 0.
        loss += lfn(X['HR'], Y['HR'])
        loss += lfn(X['HR_S'], Y['HR_S'])
        loss += lfn(X['HR_D'], Y['HR_D'])
        return loss

    if params.net == "keyvsrc":
        return lambda X, Y: CharbonnierLoss()(X['HR_lab'], Y['HR_lab'])

def batch_to_device(batch, device):
    device_batch = {k: Variable(batch[k]).to(device, non_blocking=True) for k in batch}
    return device_batch

def get_item(target, lr, key, params, train=False, noise_fn=None):
    if params.net == "keyvsrc":
        key_frame_int = params.train_key_frame_int if train else params.eval_key_frame_int
        train_item, target = keyvsrc_module.get_item(
            target, lr, key, params.upscale_factor, key_frame_int,
            params.downsampling_method, noise_fn)
    else:
        sys.exit("Net %s doesn't exist" % params.net)

    return train_item, target

def write_outputs(batch, output, target, sample_ids, output_dir, params, file_fmt):
    key_frames = True if (params.net == "keyvsrc") else False

    lr = batch['LR']
    if key_frames:
        key = batch['key']
        key_frame_int = batch['key_frame_int'][0]
    target = target['HR']
    if params.net == "keyvsrc":
        key = utils.tensor_lab2rgb(key)
        output = utils.tensor_lab2rgb(output['HR_lab'])
    else:
        output = output['HR']

    # Crop border
    if params.reflect_padding == 1:
        lr_pad = 2
        hr_pad = 2 * params.upscale_factor
        lr = lr[:, :, :, lr_pad:-lr_pad, lr_pad:-lr_pad]
        output = output[:, :, :, hr_pad:-hr_pad, hr_pad:-hr_pad]
        target = target[:, :, :, hr_pad:-hr_pad, hr_pad:-hr_pad]
        if key_frames:
            key = key[:, :, :, hr_pad:-hr_pad, hr_pad:-hr_pad]

    utils.mkdir_if_not_exists(output_dir)
    for i, vid in enumerate(sample_ids):
        out_dirs = ['lr', 'output', 'target']
        if key_frames:
            out_dirs.append('key')

        for d in out_dirs:
            utils.mkdir_if_not_exists(os.path.join(output_dir, vid, d))

        utils.write_video(lr[i].numpy(), os.path.join(output_dir, vid, 'lr'), file_fmt)
        utils.write_video(output[i].numpy(), os.path.join(output_dir, vid, 'output'), file_fmt)
        utils.write_video(target[i].numpy(), os.path.join(output_dir, vid, 'target'), file_fmt)
        if key_frames:
            utils.write_video(key[i].numpy(), os.path.join(output_dir, vid, 'key'),
                              file_fmt, step=key_frame_int)
