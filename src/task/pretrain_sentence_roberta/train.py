# coding=utf-8
# Copyright 2021 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import time
import argparse
import os
import h5py

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.cuda.amp as amp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaForMaskedLM, PretrainedConfig

from task.pretrain_sentence_roberta.data_source import DataSource, collate_fn
from task.helpers import StatisticsReporter
from optimization.lr_scheduler import get_linear_schedule_with_warmup
from task.pretrain_sentence_roberta.sentence_roberta import SentenceRoberta
from corpus.sentence_book.config import Config as SentenceBookConfig

TASK = "pretrain_roberta"


def str2bool(v):
    return v.lower() in ('true', '1', "True")


def mp_print(text, rank):
    if rank == 0:
        print(text)


def load_docs_from_filepath(filepath):
    docs = os.listdir(filepath)
    return docs




def forward_step(model, batch_data):
    data_dict, correct_tokens, masked_tokens = batch_data
    data_dict['input_embeds'] = data_dict['input_embeds'].to(model.device)
    data_dict['position_ids'] = data_dict['position_ids'].to(model.device)
    data_dict['attn_masks'] = data_dict['attn_masks'].to(model.device)
    # forward
    model_outputs = model(
        data_dict
    )
    loss = F.mse_loss(model_outputs[masked_tokens], correct_tokens[masked_tokens])
    ppl = None

    return loss, ppl


def train(local_rank, config):
    global_rank = config.node_rank * config.n_gpus + local_rank
    print(f"local rank: {[local_rank]}, global_rank: {[global_rank]}")

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # multi-gpu init
    if torch.cuda.is_available():
        if config.world_size > 1:
            dist.init_process_group(                                   
                backend='nccl',                                    
                init_method='env://',                                   
                world_size=config.world_size,                              
                rank=global_rank                                 
            )
            torch.cuda.set_device(local_rank)
            DEVICE = torch.device("cuda", local_rank)
        else:
            DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


    # build data source and reporters
    trn_reporter = StatisticsReporter()
    dev_reporter = StatisticsReporter()

    sentence_book_config = SentenceBookConfig()

    training_files = [k for k in h5py.File('../data/sentence-book/dataset.hdf5',mode='r')['train'].keys()]
    dev_files =  [k for k in h5py.File('../data/sentence-book/dataset.hdf5',mode='r')['val'].keys() ] 

    mp_print(f"Number of training files: {len(training_files)}", global_rank)
    mp_print(f"Number of dev files: {len(dev_files)}", global_rank)

    # load dev data
    if global_rank == 0:
        dev_docs = dev_files
        

        mp_print("----- Loading dev data -----", global_rank)
        dev_data_source = DataSource(config,dev_docs, "val")
        mp_print(str(dev_data_source.statistics), global_rank)
        dev_dataloader = torch.utils.data.DataLoader(
            dev_data_source,
            batch_size=config.eval_batch_size,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=True
        )

    # build model
    model_config = PretrainedConfig.from_json_file(config.model_config_filepath)
    model = SentenceRoberta(config.num_sentence_embedding_dim,model_config)
    model = model.to(DEVICE)

    # load model from checkpoint
    if config.checkpoint_path:
        mp_print("----- Checkpoint loaded -----", global_rank)
        mp_print("checkpoint path: {}".format(config.checkpoint_path), global_rank)
        checkpoint = torch.load(config.checkpoint_path, map_location=model.device)
        mp_print("loading model state dict...", global_rank)
        model.load_state_dict(checkpoint["model"])
        model.tie_weights()  # NOTE: don't forget to tie weights after loading weights

    # use mixed precision
    if config.use_amp:
        scaler = amp.GradScaler()

    # use multi gpus
    if config.world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    # build optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.l2_penalty},
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.init_lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=config.l2_penalty
    )

    # build lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.n_warmup_steps,
        num_training_steps=config.n_training_steps,
    )
    
    # init environment or load from checkpoint
    if config.checkpoint_path:
        if config.resume_training:
            mp_print("loading optimizer state dict...", global_rank)
            optimizer.load_state_dict(checkpoint["optimizer"])
            mp_print("recovering lr scheduler...", global_rank)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            mp_print("recovering others...", global_rank)
            n_step = checkpoint["n_step"]
            start_n_epoch = checkpoint["n_epoch"]
            start_train_file_idx = checkpoint["start_train_file_idx"]
            best_loss = checkpoint.get("best_loss", float("inf"))
        else:
            n_step = 0
            start_n_epoch = 0
            start_train_file_idx = 0
            best_loss = float("inf")
        OUTPUT_FILEID = checkpoint["output_fileid"]
        del checkpoint
        torch.cuda.empty_cache()
    else:
        n_step = 0
        start_n_epoch = 0
        start_train_file_idx = 0
        best_loss = float("inf")

        # names
        OUTPUT_FILEID = "roberta-ja-{}.seed_{}.{}".format(
            config.model_size,
            config.seed,
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        )
    if config.filename_note:
        OUTPUT_FILEID += f".{config.filename_note}"
    
    # define logger
    def mlog(s):
        if global_rank == 0:
            if config.enable_log:
                if not os.path.exists(f"../log/{TASK}"):
                    os.makedirs(f"../log/{TASK}")
                with open(f"../log/{TASK}/{OUTPUT_FILEID}.log", "a+", encoding="utf-8") as log_f:
                    log_f.write(s+"\n")
            mp_print(s, global_rank)
    if config.enable_log:
        if global_rank == 0:
            tb_writer = SummaryWriter(
                log_dir=f"../log/{TASK}/{OUTPUT_FILEID}",
                max_queue=5
            )

    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----")
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v))

    for epoch_idx in range(start_n_epoch, config.n_epochs):
        for train_file_idx in range(start_train_file_idx, len(training_files), config.n_train_files_per_group):
            
            train_docs = training_files[train_file_idx:train_file_idx+config.n_train_files_per_group]
            
            train_data_source = DataSource(config, train_docs, "train")
            mp_print(str(train_data_source.statistics), global_rank)
            # single gpu or cpu
            if config.world_size == 1 or not torch.cuda.is_available():
                train_data_sampler = RandomSampler(
                    train_data_source,
                    replacement=False
                )
                train_dataloader = torch.utils.data.DataLoader(
                    train_data_source,
                    batch_size=config.batch_size,
                    sampler=train_data_sampler,
                    num_workers=8,
                    collate_fn=collate_fn,
                    pin_memory=True
                )
            # multi gpus
            else:
                train_data_sampler = DistributedSampler(
                    train_data_source,
                    num_replicas=config.world_size,
                    rank=global_rank
                )
                train_dataloader = torch.utils.data.DataLoader(
                    train_data_source,
                    batch_size=config.batch_size,
                    sampler=train_data_sampler,
                    num_workers=8,
                    collate_fn=collate_fn,
                    pin_memory=True
                )

            if isinstance(train_data_sampler, DistributedSampler):
                train_data_sampler.set_epoch(epoch_idx)

            for batch_data in train_dataloader:
                n_step += 1

                # stop if reaches the maximum tranining step
                if n_step >= config.n_training_steps:
                    break

                # forward
                model.train()
                with amp.autocast():
                    loss, ppl = forward_step(model,train_data_source, batch_data, config.mask_prob)

                # update statisitcs
                trn_reporter.update_data({ "loss": loss.item()})

                # backward
                loss /= config.n_accum_steps
                if config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                del loss

                if n_step % config.n_accum_steps == 0:
                    # clip gradient
                    if config.max_grad_norm > 0.0:
                        if config.use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                    # update model parameters
                    if config.use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # zero gradients
                    optimizer.zero_grad()

                # check loss
                if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                    lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
                    log_s = f"{time.time()-start_time:.2f}s Epoch {epoch_idx}, step {n_step}, lr {lr:.5g} - "
                    log_s += trn_reporter.to_string()
                    mlog(log_s)

                    if config.enable_log and global_rank == 0:
                        for k, v in trn_reporter.items():
                            tb_writer.add_scalar(f"{k}/train", np.mean(v), n_step)

                    trn_reporter.clear()

                # evaluation on dev dataset
                if global_rank == 0 and n_step > 0 and n_step % config.validate_after_n_step == 0:
                    
                    # forward
                    with torch.no_grad():
                        model.eval()
                        
                        # use only 1 gpu for evaluation in multi-gpu situation
                        if config.world_size > 1:
                            eval_model = model.module
                        else:
                            eval_model = model

                        for eval_batch_idx, eval_batch_data in enumerate(dev_dataloader):
                            with amp.autocast():
                                loss, ppl = forward_step(eval_model,dev_data_source, eval_batch_data, config.mask_prob)
                            dev_reporter.update_data({ "loss": loss.item()})

                            if eval_batch_idx == len(dev_dataloader) - 1:
                                break

                    log_s = f"\n<Dev> - {time.time()-start_time:.3f}s - "
                    log_s += dev_reporter.to_string()
                    mlog(log_s)

                    # Save model if it has better monitor measurement
                    if config.save_model:
                        if not os.path.exists(f"../data/model/{TASK}"):
                            os.makedirs(f"../data/model/{TASK}")

                        model_to_save = model.module if hasattr(model, 'module') else model

                        # save current model
                        checkpoint = {
                            "model": model_to_save.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "n_epoch": epoch_idx,
                            "n_step": n_step,
                            "start_train_file_idx": train_file_idx,
                            "output_fileid": OUTPUT_FILEID,
                        }
                        torch.save(
                            checkpoint,
                            f"../data/model/{TASK}/{OUTPUT_FILEID}.checkpoint"
                        )
                        mlog(f"checkpoint saved to data/model/{TASK}/{OUTPUT_FILEID}.checkpoint")

                        # save best model
                        cur_loss = dev_reporter.get_value("loss")
                        if cur_loss < best_loss:
                            best_loss = cur_loss

                            torch.save(
                                checkpoint,
                                f"../data/model/{TASK}/{OUTPUT_FILEID}.best.checkpoint"
                            )
                            mlog(f"best checkpoint saved to data/model/{TASK}/{OUTPUT_FILEID}.best.checkpoint")

                    if config.enable_log:
                        for k, v in dev_reporter.items():
                            tb_writer.add_scalar(f"{k}/dev", np.mean(v), n_step)

                    dev_reporter.clear()
                    torch.cuda.empty_cache()

                # decay learning rate
                lr_scheduler.step()

        # reset starting training file index for every epoch (if might be set to a larger value if resuming from a checkpoint)
        start_train_file_idx = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # modeling
    parser.add_argument("--model_size", type=str, default="base", help="for naming")
    parser.add_argument("--model_config_filepath", type=str, default="model/roberta-ja-base-config.json", help="path to model config file")
    
    # training
    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training. 32 for base.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="batch size for evaluation")
    parser.add_argument("--n_train_files_per_group", type=int, default=1000, help="number of files to load for every loading")
    parser.add_argument("--n_training_steps", type=int, default=3e6, help="number of maximum training steps. 3e6 for base.")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of maximum training epochs")
    parser.add_argument("--n_warmup_steps", type=int, default=1e4, help="number of warmup steps. 1e4 for base.")
    parser.add_argument("--balanced_corpora", type=str, help="use the same number of files for each training corpus when there are multiple corpora. In [None, 'undersample', 'oversample', 'custom_ratio'].")
    parser.add_argument("--small_data", type=str2bool, default=False, help="use a small portion of data for bugging")
    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum input sequence length")
    parser.add_argument("--n_accum_steps", type=int, default=16, help="number of gradient accumulation steps. 16 for base.")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="probability of masking a token")

    # multi-gpu
    parser.add_argument("--n_nodes", type=int, default=1, help="number of nodes; See pytorch DDP tutorial for details")
    parser.add_argument("--n_gpus", type=int, default=1, help="number of GPUs; See pytorch DDP tutorial for details")
    parser.add_argument("--node_rank", type=int, default=0, help="rank of starting node; See pytorch DDP tutorial for details")
    parser.add_argument("--master_port", type=str, default="12321", help="port of starting node; See pytorch DDP tutorial for details")

    # mixed precision
    parser.add_argument("--use_amp", type=str2bool, default=True, help="use mixed precision for training")

    # optimizer
    parser.add_argument("--l2_penalty", type=float, default=0.01, help="l2 penalty")
    parser.add_argument("--init_lr", type=float, default=6e-4, help="peak learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping threshold")

    # management
    parser.add_argument("--corpora", type=str, nargs="+", default=["jp_cc100", "jp_wiki"], help="training corpora")
    parser.add_argument("--checkpoint_path", help="path to saved checkpoint file")
    parser.add_argument("--resume_training", type=str2bool, default=False, help="resume training from checkpoint or not")
    parser.add_argument("--enable_log", type=str2bool, default=False, help="save training log or not")
    parser.add_argument("--save_model", type=str2bool, default=False, help="save model to checkpoint or not")
    parser.add_argument("--check_loss_after_n_step", type=int, default=1e2, help="print loss after every this number of steps")
    parser.add_argument("--validate_after_n_step", type=int, default=5e3, help="validate model after every this number of steps")
    parser.add_argument("--filename_note", type=str, help="suffix of saved files' names")
    parser.add_argument("--num_sentence_embedding_dim", type=int, default=768, help="number of sentence embedding dimension")

    config = parser.parse_args()

    # multi-gpu config
    config.world_size = config.n_gpus * config.n_nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.master_port

    # run multi-processes
    if config.world_size > 1:
        mp.spawn(train, nprocs=config.n_gpus, args=(config,))
    else:
        train(0, config)
