# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import logging
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import data_functions
import loss_functions
import models


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='Rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='Number of processes, do not set! Done by multiproc module')

    return parser


def add_args(args):
    """ Train hyperparameters.
    """

    args.output = r'.\output'           # Directory to save checkpoints
    args.dataset_path = r'.\data'       # Path to dataset
    args.model_name = 'Tacotron2'       # Model to train
    args.log_file = 'train.log'         # Filename for logging
    args.epochs = 1501                  # Number of total epochs to run
    args.gpu_run = torch.cuda.is_available()   # Run on CPU or GPU

    # Epochs after which decrease learning rate
    args.anneal_steps = [10, 15]

    args.anneal_factor = 0.1           # Factor for annealing learning rate

    args.seed = None                   # Seed for random number generators

    # training
    args.epochs_per_checkpoint = 5     # Number of epochs per checkpoint
    args.checkpoint_path = ''          # Checkpoint path to resume training

    # Resumes training from the last checkpoint;
    # uses the directory provided with \'--output\' option
    # to search for the checkpoint \"checkpoint_<model_name>_last.pt\"')
    args.resume_from_last = True

    args.dynamic_loss_scaling = True    # Enable dynamic loss scaling
    args.amp = True                     # Enable AMP
    args.cudnn_enabled = True           # Enable cudnn
    args.cudnn_benchmark = True         # Run cudnn benchmark

    # disable uniform initialization of batchnorm layer weight
    args.disable_uniform_initialize_bn_weight = True

    # Optimization parameters
    args.use_saved_learning_rate = False
    args.learning_rate = 0.001         # Learing rate
    args.weight_decay = 1e-6           # Weight decay
    args.grad_clip_thresh = 1.0        # Clip threshold for gradients
    args.batch_size = 3                # Batch size per GPU

    # Size of dataset to use for train/test
    # args.subset_size = args.batch_size * 80
    args.subset_size = None

    # args.subset_size = None          # Size of dataset to use for train/test

    # Enables gradient clipping and sets maximum gradient norm value
    args.grad_clip = 5.0

    # dataset parameters

    # Loads mel spectrograms from disk instead of computing them on the fly
    args.load_mel_from_disk = True

    # Path to training filelist
    args.training_files = 'train-clean-100'

    # Path to validation filelist
    args.validation_files = 'test-clean'

    # Type of text cleaners for input text
    args.text_cleaners = ['english_cleaners']

    # audio parameters

    args.sampling_rate = 16000         # Sampling rate
    args.filter_length = 1024          # Filter length
    args.hop_length = 256              # Hop (stride) length
    args.win_length = 1024             # Window length
    args.mel_fmin = 0.0                # Minimum mel frequency
    args.mel_fmax = 8000.0             # Maximum mel frequency

    # Url used to set up distributed training
    args.dist_url = 'tcp://localhost:23456'

    args.group_name = 'group_name'     # Distributed group name
    args.dist_backend = 'gloo'         # Distributed run backend

    args.bench_class = ''

    return args


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / num_gpus
    else:
        rt = torch.div(rt, num_gpus, rounding_mode='floor')
    return rt


def init_distributed(args, world_size, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=world_size, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def save_checkpoint(model, optimizer, scaler, epoch, config, output_dir,
                    model_name, local_rank, world_size):

    random_rng_state = torch.random.get_rng_state().cuda()
    cuda_rng_state = torch.cuda.get_rng_state(local_rank).cuda()

    random_rng_states_all = [torch.empty_like(
        random_rng_state) for _ in range(world_size)]
    cuda_rng_states_all = [torch.empty_like(
        cuda_rng_state) for _ in range(world_size)]

    if world_size > 1:
        dist.all_gather(random_rng_states_all, random_rng_state)
        dist.all_gather(cuda_rng_states_all, cuda_rng_state)
    else:
        random_rng_states_all = [random_rng_state]
        cuda_rng_states_all = [cuda_rng_state]

    random_rng_states_all = torch.stack(random_rng_states_all).cpu()
    cuda_rng_states_all = torch.stack(cuda_rng_states_all).cpu()

    if local_rank == 0:
        checkpoint = {'epoch': epoch,
                      'cuda_rng_state_all': cuda_rng_states_all,
                      'random_rng_states_all': random_rng_states_all,
                      'config': config,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scaler': scaler.state_dict()}

        checkpoint_filename = f"checkpoint_{model_name}_{epoch}.pt"
        checkpoint_path = os.path.join(output_dir, checkpoint_filename)
        print(
            f"Saving model and optimizer state at epoch {epoch} to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

        symlink_dst = os.path.join(
            output_dir, f"checkpoint_{model_name}_last.pt")
        torch.save(checkpoint, symlink_dst)


def get_last_checkpoint_filename(output_dir, model_name):
    last = os.path.join(output_dir, f"checkpoint_{model_name}_last.pt")
    if os.path.exists(last):
        print("Loading checkpoint from ", last)
        return last
    else:
        print("No last checkpoint available - starting from epoch 0 ")
        return ""


def load_checkpoint(model, optimizer, scaler, epoch, filepath, local_rank):

    checkpoint = torch.load(filepath, map_location='cpu')

    epoch[0] = checkpoint['epoch'] + 1
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state_all'][device_id])
    if 'random_rng_states_all' in checkpoint:
        torch.random.set_rng_state(
            checkpoint['random_rng_states_all'][device_id])
    elif 'random_rng_state' in checkpoint:
        torch.random.set_rng_state(checkpoint['random_rng_state'])
    else:
        raise Exception(
            "Model checkpoint must have either 'random_rng_state' or 'random_rng_states_all' key.")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    return checkpoint['config']


# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license

@contextmanager
def evaluating(model):
    '''Temporarily switch to evaluation mode.'''
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def validate(model, criterion, valset, epoch, batch_iter, batch_size,
             world_size, collate_fn, distributed_run, perf_bench,
             batch_to_gpu, amp_run, args, logger):
    """Handles all the validation scoring and printing
    """
    with evaluating(model), torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=1, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn,
                                drop_last=(True if perf_bench else False))

        val_loss = 0.0
        num_iters = 0
        val_items_per_sec = 0.0
        for i, batch in enumerate(val_loader):
            if args.gpu_run:
                torch.cuda.synchronize()
            iter_start_time = time.perf_counter()

            x, y, num_items = batch_to_gpu(batch)
            # AMP upstream autocast
            with torch.cuda.amp.autocast(enabled=amp_run):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, world_size).item()
                reduced_num_items = reduce_tensor(num_items.data, 1).item()
            else:
                reduced_val_loss = loss.item()
                reduced_num_items = num_items.item()
            val_loss += reduced_val_loss
            if args.gpu_run:
                torch.cuda.synchronize()
            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time

            items_per_sec = reduced_num_items/iter_time
            logger.debug(dict(step=(epoch, batch_iter, i), data={
                         'val_items_per_sec': items_per_sec}))
            val_items_per_sec += items_per_sec
            num_iters += 1

        val_loss = val_loss / num_iters
        val_items_per_sec = val_items_per_sec / num_iters

        logger.debug(dict(step=(epoch,), data={'val_loss': val_loss}))
        logger.debug(dict(step=(epoch,), data={
                     'val_items_per_sec': val_items_per_sec}))

        return val_loss, val_items_per_sec


def adjust_learning_rate(iteration, epoch, optimizer, learning_rate,
                         anneal_steps, anneal_factor, rank, logger):
    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p += 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor ** p)

    if optimizer.param_groups[0]['lr'] != lr:
        logger.debug(dict(step=(epoch, iteration), data={'learning_rate changed': str(
            optimizer.param_groups[0]['lr'])+" -> "+str(lr)}))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Args:
    def __init__(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    part_args, _ = parser.parse_known_args()
    args = Args()
    args.rank = part_args.rank
    args.world_size = part_args.world_size
    args = add_args(args)

    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = args.rank
        world_size = args.world_size

    distributed_run = world_size > 1

    if args.seed is not None:
        torch.manual_seed(args.seed + local_rank)
        np.random.seed(args.seed + local_rank)

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    file_handler = logging.FileHandler(
        rf'{args.output}\{args.model_name}_{local_rank}_{args.log_file}', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler_format = '%(asctime)s | %(levelname)s | %(lineno)d: %(message)s'
    file_handler.setFormatter(logging.Formatter(file_handler_format))
    logger.addHandler(file_handler)

    for k, v in vars(args).items():
        logger.info(dict(step="PARAMETER", data={k: v}))
    logger.info(dict(step="PARAMETER", data={'model_name': 'Tacotron2_PyT'}))

    logger.debug('run_time', {'unit': 's'})
    logger.debug('val_loss', {'unit': None})
    logger.debug('train_items_per_sec', {'unit': 'items/s'})
    logger.debug('val_items_per_sec', {'unit': 'items/s'})

    model_name = args.model_name
    args = models.model_parser(model_name, args)

    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if distributed_run:
        init_distributed(args, world_size, local_rank, args.group_name)

    if args.gpu_run:
        torch.cuda.synchronize()
    run_start_time = time.perf_counter()

    model_config = models.get_model_config(model_name, args)
    model = models.get_model(model_name, model_config,
                             cpu_run=not args.gpu_run,
                             uniform_initialize_bn_weight=not args.disable_uniform_initialize_bn_weight)

    if distributed_run:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        sigma = args.sigma
    except AttributeError:
        sigma = None

    start_epoch = [0]

    if args.resume_from_last:
        args.checkpoint_path = get_last_checkpoint_filename(
            args.output, model_name)

    if args.checkpoint_path != "":
        model_config = load_checkpoint(model, optimizer, scaler, start_epoch,
                                       args.checkpoint_path, local_rank)

    start_epoch = start_epoch[0]

    criterion = loss_functions.get_loss_function(model_name, sigma)

    try:
        n_frames_per_step = args.n_frames_per_step
    except AttributeError:
        n_frames_per_step = None

    collate_fn = data_functions.get_collate_function(
        model_name, n_frames_per_step)
    trainset = data_functions.get_data_loader(
        model_name, args.dataset_path, args.training_files, args)
    if distributed_run:
        train_sampler = DistributedSampler(trainset, seed=(args.seed or 0))
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=args.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)

    valset = data_functions.get_data_loader(
        model_name, args.dataset_path, args.validation_files, args)

    batch_to_gpu = data_functions.get_batch_to_gpu(model_name)

    iteration = 0
    train_epoch_items_per_sec = 0.0
    val_loss = 0.0
    num_iters = 0

    model.train()

    for epoch in range(start_epoch, args.epochs):
        if args.gpu_run:
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()
        # used to calculate avg items/sec over epoch
        reduced_num_items_epoch = 0

        train_epoch_items_per_sec = 0.0

        num_iters = 0
        reduced_loss = 0

        if distributed_run:
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if args.gpu_run:
                torch.cuda.synchronize()
            iter_start_time = time.perf_counter()
            logger.info(dict(step=(epoch, i), data={
                        'glob_iter/iters_per_epoch': f'{iteration}/{len(train_loader)}'}))
            adjust_learning_rate(iteration, epoch, optimizer, args.learning_rate,
                                 args.anneal_steps, args.anneal_factor, local_rank, logger)

            model.zero_grad()
            x, y, num_items = batch_to_gpu(batch)

            # AMP upstream autocast
            with torch.cuda.amp.autocast(enabled=args.amp):
                y_pred = model(x)
                loss = criterion(y_pred, y)

            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, world_size).item()
                reduced_num_items = reduce_tensor(num_items.data, 1).item()
            else:
                reduced_loss = loss.item()
                reduced_num_items = num_items.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            logger.info(dict(step=(epoch, i), data={
                        'train_loss': reduced_loss}))

            num_iters += 1

            # accumulate number of items processed in this epoch
            reduced_num_items_epoch += reduced_num_items

            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)
                optimizer.step()

            model.zero_grad(set_to_none=True)

            if args.gpu_run:
                torch.cuda.synchronize()
            iter_stop_time = time.perf_counter()
            iter_time = iter_stop_time - iter_start_time
            items_per_sec = reduced_num_items/iter_time
            train_epoch_items_per_sec += items_per_sec

            logger.info(dict(step=(epoch, i), data={
                             'train_items_per_sec': items_per_sec}))
            logger.info(dict(step=(epoch, i), data={
                        'train_iter_time': iter_time}))
            iteration += 1

        if args.gpu_run:
            torch.cuda.synchronize()
        epoch_stop_time = time.perf_counter()
        epoch_time = epoch_stop_time - epoch_start_time

        logger.info(dict(step=(epoch,), data={'train_items_per_sec':
                         (train_epoch_items_per_sec/num_iters if num_iters > 0 else 0.0)}))
        logger.info(dict(step=(epoch,), data={'train_loss': reduced_loss}))
        logger.info(dict(step=(epoch,), data={'train_epoch_time': epoch_time}))

        val_loss, val_items_per_sec = validate(model, criterion, valset, epoch,
                                               iteration, args.batch_size,
                                               world_size, collate_fn,
                                               distributed_run,
                                               args.bench_class == "perf-train",
                                               batch_to_gpu,
                                               args.amp, args, logger)

        if ((epoch % args.epochs_per_checkpoint == 0) and
                (args.bench_class == "" or args.bench_class == "train")):
            save_checkpoint(model, optimizer, scaler, epoch, model_config,
                            args.output, args.model_name, local_rank, world_size)
        if local_rank == 0:
            file_handler.flush()

    if args.gpu_run:
        torch.cuda.synchronize()
    run_stop_time = time.perf_counter()
    run_time = run_stop_time - run_start_time
    logger.info(dict(step=tuple(), data={'run_time': run_time}))
    logger.info(dict(step=tuple(), data={'val_loss': val_loss}))
    logger.info(dict(step=tuple(), data={'train_loss': reduced_loss}))
    logger.info(dict(step=tuple(), data={'train_items_per_sec':
                     (train_epoch_items_per_sec/num_iters if num_iters > 0 else 0.0)}))
    logger.info(dict(step=tuple(), data={
                'val_items_per_sec': val_items_per_sec}))

    if local_rank == 0:
        file_handler.flush()


if __name__ == '__main__':
    main()