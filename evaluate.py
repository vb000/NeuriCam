"""Evaluates the model"""

import argparse
import logging
import os
import random

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.profiler import profile, record_function, ProfilerActivity

import utils
import model.net as net
import model.dataset as dataset

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', default=None,
                    help="Directory containing the val target set")
parser.add_argument('--lr_dir', default=None,
                    help="Directory containing the val lr set")
parser.add_argument('--key_dir', default=None,
                    help="Directory containing the val key set")
parser.add_argument('--file_fmt', default='frame%d.png',
                    help="Dataset file fmt")
parser.add_argument('--model_dir', default='experiments/keyvsrc_attn',
                    help="Directory containing params.json")
parser.add_argument('--num_steps', default=None, type=int,
                    help="Number of batches to evaluate on. Full dataset when set to None.")
parser.add_argument('--eval_batch_size', default=1,
                    help="Batch size for evaluation.")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--output_dir', default=None,
                    help="Directory containing the val lr set")
parser.add_argument('--profile', default=0, type=int,
                    help="Log profiling information.")
parser.add_argument('--use_cpu', default=0, type=int,
                    help="Use CPU even when GPUs are available.")
parser.add_argument('--gpu', default=0, type=int,
                    help="Use CPU even when GPUs are available.")
parser.add_argument('--data_parallel', default=0, type=int,
                    help="Use data parallel for multi-gpu inference")

def evaluate(model, loss_fn, dataloader, params, metrics, num_steps=None, output_dir=None, file_fmt="frmae%d.png",
             profiling=0):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    if output_dir:
        logging.info("Writing results to %s..." % output_dir)

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = {k: [] for k in metrics.metrics + ['loss', 'runtime_per_batch', 'sample_ids']}

    with torch.no_grad():
        # compute metrics over the dataset
        for i, (train_batch, target, sample_ids) in enumerate(tqdm(dataloader)):
            if (num_steps is not None) and (i == num_steps):
                break

            train_batch = net.batch_to_device(train_batch, params.device)
            target = net.batch_to_device(target, params.device)
 
            # compute model output and runtime
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    output_batch = model(train_batch)
            if profiling:
                logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            summ['runtime_per_batch'].append(prof.profiler.self_cpu_time_total/1000)

            # Compute loss
            loss = loss_fn(output_batch, target)
            
            train_batch = {k: train_batch[k].data.cpu() for k in train_batch}
            output_batch = {k: output_batch[k].data.cpu() for k in output_batch}
            target = {k: target[k].data.cpu() for k in target}
            
            # compute all metrics on this batch
            metrics_batch = metrics(output_batch, target)
            for metric in metrics_batch:
                summ[metric] += metrics_batch[metric]
            summ['loss'].append(loss.item())
            summ['sample_ids'] += sample_ids

            if output_dir:
                net.write_outputs(train_batch, output_batch, target, sample_ids,
                                  output_dir, params, file_fmt)

    # compute mean of all metrics in summary
    for i, sample in enumerate(summ['sample_ids']):
        sample_metrics = {metric: summ[metric][i]
                          for metric in metrics.metrics + ['runtime_per_batch']}
        sample_metrics_string = " ; ".join("{}: {:06.4f}".format(k, v)
                                for k, v in sample_metrics.items())
        logging.info("%s : " % sample + sample_metrics_string)

    # Remove first batch's runtime before computing mean, to account for warmup
    summ['runtime_per_batch'] = summ['runtime_per_batch'][1:]

    mean_metrics = {metric: np.mean(summ[metric])
                    for metric in metrics.metrics + ['loss']}
    mean_metrics_string = " ; ".join("{}: {:06.4f}".format(k, v)
                          for k, v in mean_metrics.items())
    logging.info("- Mean metrics : " + mean_metrics_string)
    return mean_metrics, summ


if __name__ == '__main__':
    """
    Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.device = torch.device(
        "cuda:%d" % args.gpu if (torch.cuda.is_available() and args.use_cpu==0) else "cpu")
    params.data_parallel = True if args.data_parallel != 0 else False

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    random.seed(230)
    np.random.seed(230)
    if params.device.type == 'cuda':
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    params.eval_batch_size = args.eval_batch_size
    test_dl = dataset.get_dataloader(args.target_dir, args.lr_dir, args.key_dir,
                                     params, frame_fmt=args.file_fmt, train=False)

    logging.info("- done.")

    # Define the model
    model = net.Net(params)
    if params.data_parallel:
        logging.info('Using data parallel model')
        model = nn.DataParallel(model)
    model.to(params.device)
    logging.info("Evaluating %s" % params.net)

    # Metric
    metrics = net.Metrics(params)

    # fetch loss function and metrics
    loss_fn = net.loss_fn(params)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    if args.restore_file:
        utils.load_checkpoint(os.path.join(
            args.model_dir, args.restore_file + '.pth.tar'), model, data_parallel=params.data_parallel)

    # Evaluate
    test_metrics, _ = evaluate(model, loss_fn, test_dl, params, metrics, args.num_steps,
                               args.output_dir, args.file_fmt, args.profile)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
