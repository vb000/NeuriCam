# [NeuriCam: Video Super Resolution and Colorization Using Key Frames](https://arxiv.org/pdf/2207.12496.pdf) (Accepted at MobiCom 2023)

A system based on key-frame video super-reosolution and colorization to achieve low-power
video capture from dual-mode IOT cameras. This repository holds code for the model, NeuriCam-net,
that runs on an edge receiver. NeuriCam-net reconstructs a high-reoslution color video
from low-resolution grayscale stream, using periodic high-resolution key-frames.

<p align="center">
  <img src="model/demo.gif" />
</p>


## System overview

Our dual mode IoT camera
system captures low-resolution gray-scale video from a low-
power camera and reconstructs high-resolution color frames
using the heavily duty cycled high-resolution key-frames. The
real-time neural network runs on an edge device (e.g., router)
that is not power constrained.

<p align="center">
  <img width=70% src="model/system.png" />
</p>

## Requirements

1. Create and activate a python environment with Python 3.7 or higher:

        conda create --name neuricam python=3.7
        conda activate neuricam

2. Install requirements:

        pip install -r requirements.txt

3. `mmcv-full` is a requirement for the model. Installation of `mmcv-full` is tricky, because it needs the cuda version
to be exactly the same as the one pytorch is compiled with and needs a reasobaly old g++ (>=5.0.0, <=8.0.0). So if step
`2` fails to install `mmcv-full` successfully, you might have to run:

        CUDA_HOME=<path to cuda 10.2> pip install -r requirements.txt

## Evaluation

1. Download pretrained model ([pretrained.pth.tar](https://drive.google.com/file/d/1nrqUo_gB9IM5BbAB9epQXVkXPWAFD3gT/view?usp=sharing))
to `experiments/bix4_keyvsrc_attn/` directory.

2. Download [spynet weights](https://drive.google.com/file/d/1guLPNOS8FwCye6PZ-yKL6brWJqrLaWhK/view?usp=sharing) to `model/keyvsrc`.

3. Evaluate:

        python evaluate.py --lr_dir=<path to LR dir> --key_dir=<path to Key dir> --target_dir=<ground-truth HR dir> --model_dir=experiments/bix4_keyvsrc_attn --restore_file=pretrained --file_fmt=<file format eg., "%08d.png">

## Training

1. Training the model:

        python train.py --train_lr_dir=<> --train_target_dir=<> --val_lr_dir=<> --val_target_dir=<> --model_dir=experiments/bix4_keyvsrc_attn

## Data Format

Each training or an evaluation run works on a 3 sets of videos avaialble in the following format:

```
└── lr-set                 └── key-set                └── hr-set             
    ├── my-cat-video           ├── my-cat-video           ├── my-cat-video
    │   ├── frame0.png         │   ├── frame0.png         │   ├── frame0.png
    │   ├── frame1.png         │   ├── frame15.png        │   ├── frame1.png
    │   ├── frame2.png         │   ├── frame30.png        │   ├── frame2.png
    │   ...                    │   ...                    │   ...
    ├── my-cat-vs-dog-video    ├── my-cat-vs-dog-video    ├── my-cat-vs-dog-video
    │   ├── frame0.png         │   ├── frame0.png         │   ├── frame0.png
    │   ├── frame1.png         │   ├── frame15.png        │   ├── frame1.png
    │   ├── frame2.png         │   ├── frame30.png        │   ├── frame2.png
    │   ...                    │   ...                    │   ...
    ...                        ...                        ...                         
```

Paths to `lr-set`, `key-set` and `hr-set` have to be provided to respective flags in the train and eval commands above.
