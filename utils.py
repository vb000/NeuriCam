import os
import sys
import glob

import json
import logging
import shutil

import numpy as np
import matplotlib.pyplot as plt

import time
import cv2
import math
from PIL import Image, ImageOps
import skimage.transform

import torch
import torch.nn as nn
import torch.nn.init as init

def psnr(img1, img2):
    assert len(img1.shape) <= 3
    assert len(img2.shape) <= 3

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)

    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def shot_noise(im, snr):
    full_well = 10.**(snr / 20.)
    photon_count = im * full_well
    poisson_photon_count = torch.poisson(photon_count)
    return poisson_photon_count / full_well

def bgr2ycbcr(img_, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    Output:
        type is same as input
        unit8, [0, 255]
        float, [0, 1]
    '''
    img = np.copy(img_)
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def rgb2normalizedLab(img_):
    # Convert to bgr format
    img = img_[:, :, ::-1]

    # Convert to L*a*b* format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Normalize the lab image to [0, 1] range
    img[:, :, 0] = img[:, :, 0] / 100.
    img[:, :, 1] = (img[:, :, 1] + 128.) / 255.
    img[:, :, 2] = (img[:, :, 2] + 128.) / 255.

    return img

def normalizedLab2rgb(img_):
    # Scale normlab image to original L*a*b* range
    img = img_.copy()
    img[:, :, 0] = img[:, :, 0] * 100.
    img[:, :, 1] = (img[:, :, 1] * 255.) - 128.
    img[:, :, 2] = (img[:, :, 2] * 255.) - 128.

    # Convert to rgb format
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def tensor_rgb2lab(rgb):
    T, C, H, W = rgb.shape

    # Convert to L*a*b* space
    lab = []
    for t in range(T):
        frm_rgb = np.transpose(rgb[t, :, :, :].numpy(), (1, 2, 0)) # [H, W, C]
        frm_lab = rgb2normalizedLab(frm_rgb)
        frm_lab = np.transpose(frm_lab, (2, 0, 1)) # [C, H, W]
        lab.append(torch.from_numpy(frm_lab))
    return torch.stack(lab, dim=0)

def tensor_lab2rgb(out):
    out = out.clone().numpy()
    out_rgb = np.zeros(out.shape)
    for i, sample in enumerate(out):
        for j, frm_lab in enumerate(sample):
            frm_lab = np.transpose(frm_lab, (1, 2, 0)) # channels last
            frm_rgb = normalizedLab2rgb(frm_lab)
            frm_rgb = np.transpose(frm_rgb, (2, 0, 1)) # channels first
            out_rgb[i, j, :, :, :] = frm_rgb
    return torch.from_numpy(out_rgb)

def get_frame_paths(video_pth, frame_fmt='frame%d.png', start_num=0, end_num=None,
                    step=1):
    frame_paths = []
    frame_idx = start_num
    while True:
        frame_pth = os.path.join(video_pth, frame_fmt % frame_idx)
        if (end_num is not None) and frame_idx >= end_num:
            break
        elif not os.path.exists(frame_pth):
            break
        else:
            frame_paths.append(frame_pth)
        frame_idx += step
    return frame_paths

def bicubic_baseline(lr, upscale_factor):
    """
    lr: [b, t, c, h, w]
    out: [b, t, c, upscale_factor*h, upscale_factor*w]
    """
    B, T, C, H, W = lr.shape
    device = lr.device
    out = []
    for sample in lr:
        s = []
        for frm in sample:
            frm = frm.permute(1, 2, 0).cpu().numpy()
            frm = cv2.resize(frm, (upscale_factor*W, upscale_factor*H),
                             interpolation=cv2.INTER_CUBIC)
            frm = torch.from_numpy(frm).to(device).permute(2, 0, 1)
            s.append(frm)
        s = torch.stack(s, dim=0)
        out.append(s)
    out = torch.stack(out, dim=0)
    return out

def model_size(model):
    return sum(p.numel() for p in model.parameters())

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def write_img(img, path):
    """
    img: numpy array in the format [C, H, W] normalized to [0, 1].
         C=1 in case of gray image, otherwise C=3 in the order RGB.
    """
    assert len(img.shape) == 3
    C, H, W = img.shape

    img_ = (255. * img.clip(0., 1.)).round().astype(np.uint8)
    img_ = img_.transpose((1, 2, 0)) # [H, W, C]
    if C == 1:
        img_ = img_.squeeze(2)
    else:
        img_ = img_[:, :, ::-1] # [H, W, BGR]

    cv2.imwrite(path, img_)

def write_video(video, path, frame_fmt="frame%d.png", start_num=0,
                end_num=None, step=1):
    assert len(video.shape) == 4
    T, C, H, W = video.shape

    if end_num is None:
        end_num = T * step

    frame_numbers = range(start_num, end_num, step)
    for frm, frm_num in zip(video, frame_numbers):
        write_img(frm, os.path.join(path, frame_fmt % frm_num))

def format_lr_info(optimizer):
    lr_info = ""
    for i, pg in enumerate(optimizer.param_groups):
        lr_info += " {group %d: params=%.5fM lr=%.1E}" % (
            i, sum([p.numel() for p in pg['params']]) / (1024 ** 2), pg['lr'])
    return lr_info

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None, data_parallel=False):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    state_dict = checkpoint['state_dict']
    if data_parallel:
        state_dict = {'module.' + k: state_dict[k] for k in state_dict}
    model.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)
