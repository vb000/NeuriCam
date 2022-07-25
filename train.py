"""Train the model"""

import argparse
import logging
import os

import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

import utils
import model.net as net
import model.dataset as dataset
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--train_target_dir', default=None,
                    help="Directory containing the train target set")
parser.add_argument('--train_lr_dir', default=None,
                    help="Directory containing the train lr set")
parser.add_argument('--train_key_dir', default=None,
                    help="Directory containing the train key set")
parser.add_argument('--val_target_dir', default=None,
                    help="Directory containing the val target set")
parser.add_argument('--val_lr_dir', default=None,
                    help="Directory containing the val lr set")
parser.add_argument('--val_key_dir', default=None,
                    help="Directory containing the val key set")
parser.add_argument('--file_fmt', default='frame%d.png',
                    help="Dataset file fmt")
parser.add_argument('--model_dir', default=None,
                    help="Directory containing params.json")
parser.add_argument('--num_steps', default=None, type=int,
                    help="Number of batches per epoch. Full dataset when set to None.")
parser.add_argument('--eval_batch_size', default=1,
                    help="Batch size for evaluation.")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--gpus', default=None, type=int,
                    help="Number of gpus to use. Default is to use all available.")
parser.add_argument('--no_restore_optim', dest='restore_optim', action='store_false',
                    help="Don't restore optimizer state when restore file is provided.")
parser.set_defaults(restore_optim=True)
parser.add_argument('--restore_only_weights', dest='restore_all', action='store_false',
                    help="Don't restore optimizer state when restore file is provided.")
parser.set_defaults(restore_all=True)

def train(model, optimizer, loss_fn, metrics, dataloader, params, num_steps):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = {k: [] for k in metrics.metrics + ['losses']}
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, target, sample_ids) in enumerate(dataloader):
            # End the epoch after `num_steps`
            if (num_steps is not None) and (i == num_steps):
                break

            train_batch = net.batch_to_device(train_batch, params.device)
            target = net.batch_to_device(target, params.device)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, target)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = {k: output_batch[k].data.cpu() for k in output_batch}
                target = {k: target[k].data.cpu() for k in target}
                
                # compute all metrics on this batch
                metrics_batch = metrics(output_batch, target)
                for metric in metrics_batch:
                    summ[metric] += metrics_batch[metric]
                summ['losses'].append(loss.item())
            
            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:06.4f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean(summ[metric])
                    for metric in metrics.metrics + ['losses']}
    metrics_string = " ; ".join("{}: {:06.4f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None, restore_all=True, num_steps=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    start_epoch = 0
    best_val_psnr = 0.0

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        if restore_all:
            logging.info("Restoring all parameters from {}".format(restore_path))
            checkpoint = utils.load_checkpoint(restore_path, model,
                    optimizer if args.restore_optim else None, params.data_parallel)
            if 'epoch' in checkpoint.keys():
                start_epoch = checkpoint['epoch']
            if 'best_val_psnr' in checkpoint.keys():
                best_val_psnr = checkpoint['best_val_psnr']
        else:
            logging.info("Restoring only weights parameters from {}".format(restore_path))
            _ = utils.load_checkpoint(restore_path, model,
                optimizer if args.restore_optim else None, params.data_parallel)
        logging.info('Learning rate after resotration:' + utils.format_lr_info(optimizer))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params.lr_patience, min_lr=1e-7,
                                                     verbose=True)

    for epoch in range(start_epoch, params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, metrics, train_dataloader, params, num_steps)

        # Evaluate for one epoch on validation set
        val_metrics, _ = evaluate(model, loss_fn, val_dataloader, params, metrics, num_steps)

        # Adjust learning rate
        if epoch >= params.fix_lr_epochs:
            scheduler.step(val_metrics['loss'])
            logging.info('Learning rate after lr scheduler step:' + utils.format_lr_info(optimizer))

        val_psnr = val_metrics[params.base_metric]
        is_best = val_psnr >= best_val_psnr

        # Save weights
        state_dict = model.state_dict()
        if params.data_parallel:
            state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict}
        utils.save_checkpoint({'epoch': epoch + 1,
                               'best_val_psnr': best_val_psnr,
                               'state_dict': state_dict,
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_psnr = val_psnr

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params.data_parallel = torch.cuda.device_count() > 1

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if params.device.type == 'cuda':
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    params.eval_batch_size = args.eval_batch_size
    train_dl = dataset.get_dataloader(args.train_target_dir, args.train_lr_dir, args.train_key_dir,
                                      params, frame_fmt=args.file_fmt, train=True)
    val_dl = dataset.get_dataloader(args.val_target_dir, args.val_lr_dir, args.val_key_dir,
                                    params, frame_fmt=args.file_fmt, train=False)

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params)
    model.to(params.device)

    # Metric
    metrics = net.Metrics(params)

    # Initialize model and get param gorups with per-group optimizer configurations
    param_groups = net.configure_parameters(model, args.model_dir, params)

    if params.data_parallel:
        logging.info('Using data parallel model.')
        device_ids = list(range(args.gpus)) if args.gpus is not None else None
        model = nn.DataParallel(model, device_ids=device_ids)
    
    # Instantiate optimizer 
    optimizer = optim.Adam(param_groups, lr=params.learning_rate,
                           betas=(0.9, 0.999), eps=1e-8)
    logging.info('Learning rates initialized to:' + utils.format_lr_info(optimizer))

    # Set LR scheduler patience
    if not 'lr_patience' in params.dict:
        params.lr_patience = 5
    logging.info("LR schedule patience set to %d epochs." % params.lr_patience)

    # fetch loss function and metrics
    loss_fn = net.loss_fn(params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file, args.restore_all, args.num_steps)
