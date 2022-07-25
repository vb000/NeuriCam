## Requirements

1. Create and activate a python environment with Python 3.7 or higher:

        python3 -m venv rkfn
        source rkfn/bin/activate

2. Install requirements:

        pip install -r requirements.txt

3. `mmcv-full` is a requirement for some models. Installation of `mmcv-full` is tricky, because it needs even for instal time,
needs cuda version to be exactly the same as the one pytorch is compiled with and needs a reasobaly old gcc with c++14
support. One example combination that worked is: `mmcv-full==1.4.1` with `torch==1.7.1+cu101`, `torchvision==0.8.2+cu101`,
`cuda 10.1` and `gcc 6.3.1`.

## Training

1. Make the [vimeo_septuplet](http://toflow.csail.mit.edu/)
dataset available in the format detailed in [Data Format](#data-format) section:

        python data/vimeo_septuplet.py --data_dir=<path to vimeo septuplet root> --out_dir=data/vimeo_septuplet
   
   This creates a video set in the [Data Format](#data-format) with simlinks to the original dataset.

2. Make validation set with 10 random vimdeo_septuplet test video sequences:
    
        cd data/vimeo_septuplet
        mkdir val && cd test && ls | sort -R 2> /dev/null | head -10 | xargs cp -rt ../val && cd ..

3. Training the model:

        python train.py --train_dir=data/vimeo_septuplet/train/ --val_dir=data/vimeo_septuplet/val/ --model_dir=experiments/rkfn_one_way/

## Evaluation

1. Download datasets ([Vid4](https://drive.google.com/file/d/1BU9OwkcKN1rVj5vvKoxikpS4XekUYz68/view?usp=sharing);
[Xiph720p](https://drive.google.com/file/d/1YerJjfyd0J55EdPpod1nmHGabSsb57Tf/view?usp=sharing);
[Xiph720p30fps](https://drive.google.com/file/d/1FmZPHAlGlCkjgKbscvVjVS8wk4-e1r8c/view?usp=sharing)) to `benchmarks` directory.

2. Download pretrained model ([pretrained.pth.tar](https://drive.google.com/file/d/1fXchMuodZ6aQc8w02NSK05Z-yVJHJRDg/view?usp=sharing))
to `experiments/rkfn_one_way/` directory.

3. Download spynet weights to `model/basicvsr_pp`.

3. Evaluate:

        python upscale_videos.py --target_dir=benchmarks/Vid4 --model_dir=experiments/rkfn_one_way/ --restore_file=pretrained --key_frame_int=15 --output_dir=Results/Vid4

For RKFN model that operates on color LR inputs use ([pretrained.pth.tar](https://drive.google.com/file/d/1OoGl95f42vvAQbCK6k1jS-l0MNwu1Z0J/view?usp=sharing))
and replace `rkfn_one_way` with `rkfn_one_way_color` in the commands above.

## Repository Structure

```
├── benchmarks              # Directory with benchmark video sets
├── data                    # Directory with train, val and test datasets
├── experiments             # Directory with experiment configurations and results
│   ├── experiment-1        # An example experiment directory
│   │   ├── best.pth.tar    # Saved weights that do best on desired metric up until this point in the training
│   │   ├── last.pth.tar    # Saved weights after last epoch up until this point in the training
│   │   ├── params.json     # Experiment configuration
│   │   ...                 # Generated logs and metrics
│   ├── experiment-2
│   │   ├── best.pth.tar
│   │   ├── last.pth.tar
│   │   ├── params.json
│   │   ...
├── model                   # Directory with network architectures and data loaders
├── evaluate.py             # Script to evaluate an experiment on a val, test or any of the benchmark video sets.
├── requirements.txt        # Required Python packages
├── train.py                # Script to train an architecture based on an experiment configuration.
├── upscale_videos.py       # The app: Generates a high-res video of an input video set.
└── utils.py                # Misc. utility functions
```

## Data Format

Each training or an evaluation run works on a set of videos avaialble in the following format:

```
│   └── my-video-set
│       ├── my-cat-video
│       │   ├── frame0.png
│       │   ├── frame1.png
│       │   ├── frame2.png
│       │   ...
│       ├── my-cat-vs-dog-video
│       │   ├── frame0.png
│       │   ├── frame1.png
│       │   ├── frame2.png
│       │   ...
│       ...
```
