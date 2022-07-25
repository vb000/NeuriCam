import os
from os import listdir
from os.path import join
import os.path
import random
import glob
import re
import sys
from random import randrange
import logging

import numpy as np
from PIL import Image, ImageOps
import cv2
from skimage import img_as_float
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(__file__))

import utils
import net

def generate_transform_params(lr_h, lr_w, reflect_padding, patch_h,
                              patch_w, scale):
    params = {}
    if reflect_padding == 1:
        params['lr_pad'] = 2
        params['hr_pad'] = 2*scale
    else:
        params['lr_pad'] = 0
        params['hr_pad'] = 0
    params['hflip'] = random.random() < 0.5
    params['vflip'] = random.random() < 0.5
    params['lr_crop'] = [random.randrange(lr_h - patch_h + 1),
                         random.randrange(lr_w - patch_w + 1),
                         patch_h, patch_w]
    params['hr_crop'] = [scale*p for p in params['lr_crop']]
    return params 

def transform(seq, pad, crop, hflip, vflip):
    """
    seq: [t, c, h, w]
    Returns: [t, c, h, w]
    """
    if crop is not None:
        seq = TF.crop(seq, *crop)
    if hflip:
        seq = TF.hflip(seq)
    if vflip:
        seq = TF.vflip(seq)
    if pad is not None:
        seq = TF.pad(seq, pad, padding_mode='reflect')
    return seq

def load_video(video_path, frame_fmt='frame%d.png', start_num=0,
               end_num=0, step=1, grayscale=False):
    frame_paths = utils.get_frame_paths(video_path, frame_fmt,
                                        start_num, end_num, step)

    video = []
    for frame_pth in frame_paths:
        if grayscale:
            frame = cv2.imread(frame_pth, cv2.IMREAD_GRAYSCALE) # [h, w, bgr]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.imread(frame_pth, cv2.IMREAD_COLOR)
        frame = frame.astype('float32') / 255. # Norm. to [0, 1]
        frame = frame[:, :, ::-1] # [h, w, rgb]
        frame = frame.transpose(2, 0, 1).copy() # [rgb, h, w]
        frame = torch.from_numpy(frame)
        video.append(frame)
    video = torch.stack(video, dim=0) # [t, c, h, w]

    return video

class KeyVSRCDataset(data.Dataset):
    def __init__(self, target_dir, lr_dir, key_dir, params,
                 frame_fmt='frame%d.png', train=False):
        super(KeyVSRCDataset, self).__init__()
        self.target_dir = target_dir
        self.lr_dir = lr_dir
        self.key_dir = key_dir
        self.frame_fmt = frame_fmt
        self.train = train
        self.params = params
        self.reflect_padding = int(params.reflect_padding)
        self.scale = int(params.upscale_factor)
        self.patch_size = int(params.patch_size)
        if train:
            self.num_frames = None if params.train_pred_per_iter == "" \
                                   else int(params.train_pred_per_iter)
            self.start_num = None if params.train_start_frame == "random" \
                                  else int(params.train_start_frame)
            self.key_frame_int = int(params.train_key_frame_int)
        else:
            self.num_frames = None if params.eval_pred_per_iter == "" \
                                   else int(params.eval_pred_per_iter)
            self.start_num = int(params.eval_start_frame)
            self.key_frame_int = int(params.eval_key_frame_int)

        # Populate video list
        if target_dir:
            self.ref_dir = target_dir
            self.videos = os.listdir(target_dir)
        elif lr_dir:
            self.ref_dir = lr_dir
            self.videos = os.listdir(lr_dir)
        else:
            print("Both target and lr directories are missing!")
            exit(1)
        self.videos.sort()

        if train and ('dataset_mult' in params.dict):
            self.videos *= params.dataset_mult

        if 'grayscale' in params.dict and params.grayscale != 0:
            self.grayscale = True
            logging.info('Converting evrything to grayscale!!!!!')
        else:
            self.grayscale = False


    def __getitem__(self, index):
        video = self.videos[index]

        # Compute start frame range from a sample video in the dataset
        sample_video_pth = os.path.join(self.ref_dir, video)
        total_num_frames = len(
            glob.glob(os.path.join(sample_video_pth,
                                   '*.' + self.frame_fmt.split('.', 1)[1])))
        num_frames = self.num_frames if self.num_frames is not None \
                                     else total_num_frames
        start_range = total_num_frames - num_frames + 1

        if self.start_num is None:
            start_num = random.choice(range(start_range)) // self.key_frame_int
            start_num *= self.key_frame_int
        else:
            start_num = self.start_num
        end_num = start_num + num_frames

        # Load videos
        if self.target_dir:
            target = load_video(os.path.join(self.target_dir, video),
                                self.frame_fmt, start_num, end_num, grayscale=self.grayscale)
            lr_h, lr_w = target.shape[2]//self.scale, target.shape[3]//self.scale
        else:
            target = None

        if self.lr_dir:
            lr = load_video(os.path.join(self.lr_dir, video),
                            self.frame_fmt, start_num, end_num, grayscale=self.grayscale)
            lr_h, lr_w = lr.shape[2:4]
        else:
            lr = None

        if self.key_dir:
            key = load_video(os.path.join(self.key_dir, video),
                             self.frame_fmt, start_num, end_num,
                             step=self.key_frame_int, grayscale=self.grayscale)
        else:
            key = None

        # Transforms
        transform_params = generate_transform_params(
            lr_h, lr_w, self.reflect_padding, self.patch_size,
            self.patch_size, self.scale)

        if self.train:
            hr_p = [transform_params[k]
                     for k in ['hr_pad', 'hr_crop', 'hflip', 'vflip']]
            lr_p = [transform_params[k] 
                     for k in ['lr_pad', 'lr_crop', 'hflip', 'vflip']]
        else:
            hr_p = (transform_params['hr_pad'], None, False, False) 
            lr_p = (transform_params['lr_pad'], None, False, False) 

        if target is not None:
            target = transform(target, *hr_p)
        if lr is not None:
            lr = transform(lr, *lr_p)
        if key is not None:
            key = transform(key, *hr_p)

        train_item, target = net.get_item(target, lr, key, self.params,
                                          self.train)
        return train_item, target, video

    def __len__(self):
        return len(self.videos)

def get_dataloader(target_dir, lr_dir, key_dir, params,
                   frame_fmt='frame%d.png', train=False):
    batch_size = int(params.batch_size) if train else int(params.eval_batch_size)
    dataset = KeyVSRCDataset(target_dir, lr_dir, key_dir, params,
                             frame_fmt, train=train)
    dl = DataLoader(dataset=dataset, num_workers=params.num_workers,
                    batch_size=batch_size, shuffle=train, pin_memory=True)
    return dl

