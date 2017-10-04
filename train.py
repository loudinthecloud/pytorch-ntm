#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import argparse
import json
import time
import random

import argcomplete
import torch
import numpy as np

from tasks.copytask import CopyTaskModelTraining, CopyTaskParams

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
}


# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    print("Using seed={}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths):
    progress_clean()

    basename = "{}/{}-{}-batch-{}".format(args.checkpoint_path, name, args.seed, batch_num)
    model_fname = basename + ".model"
    print("Saving model checkpoint to: '{}'".format(model_fname))
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    print("Saving model training history to '{}'".format(train_fname))
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def train_model(model,
                args):

    num_batches = model.params.num_batches
    batch_size = model.params.batch_size

    print("Training model for {} batches (batch_size={})...".format(
        num_batches, batch_size))

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()

    for batch_num, x, y in model.dataloader:
        loss, cost = model.train_batch(model.net, model.criterion, model.optimizer, x, y)
        losses += [loss]
        costs += [cost]
        seq_lengths += [y.size(0)]

        # Update the progress bar
        progress_bar(batch_num, args.report_interval, loss)

        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            progress_clean()
            print("Batch {} Loss: {:.6f} Cost: {:.2f} Time: {} ms/sequence".format(
                batch_num, mean_loss, mean_cost, mean_time))
            start_ms = get_ms()

        # Checkpoint
        if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
            save_checkpoint(model.net, model.params.name, args,
                            batch_num, losses, costs, seq_lengths)

    print("Done training.")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='copy',
                        help="Choose the task's model to train (default: copy)")
    parser.add_argument('--print-params', action='store_true', default=False,
                        help="Print the model's default parameters")
    parser.add_argument('--checkpoint_interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (in batches). 0 - disable")
    parser.add_argument('--checkpoint_path', action='store', default='./',
                        help="Out directory for checkpoint data (default: './')")
    parser.add_argument('--report_interval', type=int, default=REPORT_INTERVAL,
                        help="Report interval (in batches)")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def init_model(args):
    print("Training for the **{}** task".format(args.task))
    model_cls, params_cls = TASKS[args.task]
    params = params_cls()
    model = model_cls(params=params)
    return model


def main():
    # Initialize arguments
    args = init_arguments()

    if args.print_params:
        params_cls = TASKS[args.task][1]
        print(params_cls())
        return

    # Initialize random
    init_seed(args.seed)

    # Initialize the model
    model = init_model(args)

    print("Total number of parameters: {}".format(model.net.calculate_num_params()))
    train_model(model, args)


if __name__ == '__main__':
    main()
