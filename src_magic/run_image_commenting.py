# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from emotional_dataset import *
from modeling_emotional_gpt import MAGICLMHeadModel

import nltk.translate.bleu_score as bleu
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

def load_and_cache_examples(args, tokenizer, data_split='train'):
    
    dataset = NICEDatasetResnet(args, tokenizer, data_split=data_split)
    if data_split == 'train':
        n = len(dataset) % args.per_gpu_train_batch_size
    else:
        n = len(dataset) % args.per_gpu_eval_batch_size
    if n != 0:
        dataset.corpus = dataset.corpus[:-n]
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        #logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    nworkers = 16
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=nworkers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    loss_step = 0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for cur_epoch_i in train_iterator:
        epoch_loss = 0.0
        epoch_loss_step = 0
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(tqdm(train_dataloader)):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            img, liwc, inputs, labels = batch
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            img = img.unsqueeze(1).to(args.device)
            imgpos = None
            imgcls = None

            liwc = liwc.unsqueeze(1).to(args.device)

            model.train()

            loss = 0.

            for cmt_i in range(1,args.num_cmts):
                curcondition = (img, imgpos, imgcls, liwc[:,:,cmt_i,:])
                outputs = model(curcondition,inputs[:,:cmt_i*args.cmt_len],inputs[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len],labels=labels[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len])
                loss += outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            loss_step += 1
            epoch_loss += loss.item()
            epoch_loss_step += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, data_split="val")
                        for key, value in results.items():
                            # tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info("eval_{} ".format(key) + str(value) + " at step " + str(global_step))
                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)


                if global_step % 1000 == 0:
                    logger.info("Current Loss is %s", str(tr_loss / loss_step))

            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break
        epoch_eval_results = evaluate(args, model, tokenizer, data_split="val")
        epoch_eval_ppl = epoch_eval_results['perplexity']
        epoch_eval_loss = epoch_eval_results['eval_loss']
        logger.info("Epoch Training Loss is %s", str(epoch_loss / epoch_loss_step))
        logger.info("Epoch Val Loss is %s", str(epoch_eval_loss))
        logger.info("Epoch Val PPL is %s", str(epoch_eval_ppl))

        checkpoint_prefix = "checkpoint"
        # Save model checkpoint
        output_dir = os.path.join(args.epoch_output_dir, "{}-{}-{}-{}".format(checkpoint_prefix, global_step, epoch_eval_loss, epoch_eval_ppl))
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        # logger.info("Saving model checkpoint to %s", output_dir)

        _rotate_checkpoints(args, checkpoint_prefix)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="", data_split="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, data_split=data_split)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly


    eval_sampler = SequentialSampler(eval_dataset)
    nworkers = 16
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=nworkers)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch_step, batch in enumerate(tqdm(eval_dataloader)):

        img, liwc, inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        img = img.unsqueeze(1).to(args.device)
        imgpos = None
        imgcls = None

        liwc = liwc.unsqueeze(1).to(args.device)

        with torch.no_grad():
            lm_loss = 0.
            for cmt_i in range(1,args.num_cmts):
                curcondition = (img, imgpos, imgcls, liwc[:,:,cmt_i,:])
                outputs = model(curcondition, 
                    inputs[:,:cmt_i*args.cmt_len],
                    inputs[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len], 
                    labels=labels[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len])
                lm_loss += outputs[0]
            if args.n_gpu > 1:
                lm_loss = lm_loss.mean()
            eval_loss += lm_loss.item()
        nb_eval_steps += 1


    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    result = {"perplexity": perplexity, "eval_loss": eval_loss}

    return result



def generate_cmts(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="", data_split="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    model_gen = (model.module if hasattr(model, "module") else model)

    eval_output_dir = args.output_dir

    analyzer = SentimentIntensityAnalyzer()

    result={}

    eval_dataset = load_and_cache_examples(args, tokenizer,data_split=data_split)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running generation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    bleuscore1 = []
    bleuscore2 = []
    bleuscore3 = []
    bleuscore4 = []
    compd_distance = []
    sent_match = []
    nb_eval_steps = 0
    eval_gens = []
    eval_labels = []

    model_gen.eval()

    for batch_step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):

        img, liwc, inputs, labels = batch

        inputs = inputs.to(args.device)

        inputs_ids = torch.tensor(img.size(0)*[tokenizer.encode("<|endoftext|>")])
        inputs_ids = inputs_ids.to(args.device)

        img = img.unsqueeze(1).to(args.device)
        imgpos = None
        imgcls = None

        liwc = liwc.unsqueeze(1).to(args.device)

        with torch.no_grad():
            lm_logits = []

            for cmt_i in range(1, args.num_cmts):
                curcondition = (img, imgpos, imgcls, liwc[:,:,cmt_i,:])
                outputs = model(curcondition, 
                    inputs[:,:cmt_i*args.cmt_len], 
                    inputs[:,cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len],
                    max_length=args.cmt_len)
                outputs = outputs[:,1:]
                lm_logits.append(outputs)

            lm_logits = torch.cat(lm_logits, 1)
            lm_logits = lm_logits.detach().cpu().numpy()
            total_cmts = args.num_cmts-1

            for b in range(lm_logits.shape[0]):
                gen_cmts = lm_logits[b]
                cur_gen_txt = []
                cur_gen_labels = []
                for cmt_i in range(total_cmts):
                    gen_txt = list(gen_cmts[cmt_i*args.cmt_len:(cmt_i+1)*args.cmt_len])
                    if 50256 in gen_txt:
                        gen_txt = gen_txt[:gen_txt.index(50256)]
                    label = labels[b,(cmt_i+1)*args.cmt_len:(cmt_i+2)*args.cmt_len].contiguous().clone().detach().cpu().numpy()
                    label = list(label)
                    if sum(label) != -100*args.cmt_len:
                        if 50256 in label:
                            label = label[:label.index(50256)]
                        try:
                            score1 = bleu.sentence_bleu([label], gen_txt, weights=(1, 0, 0, 0))
                        except:
                            score1 = 0
                        try:
                            score2 = bleu.sentence_bleu([label], gen_txt, weights=(0.5, 0.5, 0, 0))
                        except:
                            score2 = 0
                        try:
                            score3 = bleu.sentence_bleu([label], gen_txt, weights=(0.33, 0.33, 0.33, 0))
                        except:
                            score3 = 0
                        try:
                            score4 = bleu.sentence_bleu([label], gen_txt)
                        except:
                            score4 = 0
                        bleuscore1.append(score1)
                        bleuscore2.append(score2)
                        bleuscore3.append(score3)
                        bleuscore4.append(score4)
                        gen_txt = tokenizer.decode(gen_txt)
                        label = tokenizer.decode(label)
                        gen_txt_sent = analyzer.polarity_scores(gen_txt)
                        gen_txt_sent_arr = [gen_txt_sent['neg'], gen_txt_sent['neu'], gen_txt_sent['pos']]
                        label_sent = analyzer.polarity_scores(label)
                        label_sent_arr = [label_sent['neg'], label_sent['neu'], label_sent['pos']]
                        compd_distance.append(abs(gen_txt_sent['compound'] - label_sent['compound']))
                        if np.argmax(gen_txt_sent_arr)==np.argmax(label_sent_arr):
                            sent_match.append(1)
                        else:
                            sent_match.append(0)


                        cur_gen_txt.append(gen_txt+'\t'+str(gen_txt_sent))
                        cur_gen_labels.append(label+'\t'+str(label_sent))
                eval_gens.append('\t'.join(cur_gen_txt))
                eval_labels.append('\t'.join(cur_gen_labels))

        nb_eval_steps += 1

    if data_split == 'test':
        output_gentxt_file = os.path.join(eval_output_dir, prefix, "generated_cmts.txt")
        output_labels_file = os.path.join(eval_output_dir, prefix, "labels_cmts.txt")
        output_results_file = os.path.join(eval_output_dir, prefix, "results.txt")

        with open(output_gentxt_file, 'w') as writer:
            logger.info("***** Generated Text {} *****".format(prefix))
            writer.write('\n'.join(eval_gens))

        with open(output_labels_file, 'w') as writer:
            logger.info("***** Labels Text {} *****".format(prefix))
            writer.write('\n'.join(eval_labels))

        with open(output_results_file, 'w') as writer:
            logger.info("***** Results {} *****".format(prefix))
            writer.write('Bleu-1: '+str(np.mean(bleuscore1))+'\n')
            writer.write('Bleu-2: '+str(np.mean(bleuscore2))+'\n')
            writer.write('Bleu-3: '+str(np.mean(bleuscore3))+'\n')
            writer.write('Bleu-4: '+str(np.mean(bleuscore4))+'\n')
            writer.write('Compound Distance: '+str(np.mean(compd_distance))+'\n')
            writer.write('Sentiment Match: '+str(float(np.sum(sent_match) / len(sent_match)))+'\n')

    bleu1 = str(np.mean(bleuscore1))
    bleu2 = str(np.mean(bleuscore2))
    bleu3 = str(np.mean(bleuscore3))
    bleu4 = str(np.mean(bleuscore4))
    compound_distance = str(np.mean(compd_distance))
    sentiment_match = str(float(np.sum(sent_match) / len(sent_match)))

    logger.info('Bleu-1: ' + bleu1)
    logger.info('Bleu-2: ' + bleu2)
    logger.info('Bleu-3: ' + bleu3)
    logger.info('Bleu-4: ' + bleu4)
    logger.info('Compound Distance: ' + compound_distance)
    logger.info('Sentiment Match: ' + sentiment_match)

    result = {}
    result["bleu-1"] = bleu1
    result["bleu-2"] = bleu2
    result["bleu-3"] = bleu3
    result["bleu-4"] = bleu4
    result["compound_distance"] = compound_distance
    result["sentiment_match"] = sentiment_match

    return result


# python run_image_commenting.py --data_dir /data/NICE_release/NICE --img_path /data/NICE_release/NICE/img_feat --epoch_output_dir /data/NICE_release/test --output_dir /data/NICE_release/test --model_name_or_path gpt2 --do_train 

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir", default=None, type=str, required=True, help="The path of dataset."
    )

    parser.add_argument(
        "--img_path", default=None, type=str, required=True, help="The path of image features."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getenv('PT_OUTPUT_DIR','/tmp'),
        help="The output directory where the model predictions and checkpoints will be written.",
    ) 

    parser.add_argument(
        "--epoch_output_dir",
        type=str,
        default=os.getenv('PT_OUTPUT_DIR','/tmp'),
        help="The output directory where the model predictions and checkpoints will be written.",
    ) 

    # Other parameters
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_generate", action="store_true", help="")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=10,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument("--cmt_len", type=int, default=30)
    parser.add_argument("--num_cmts", type=int, default=6)
    

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 'output')
    args.epoch_output_dir = os.path.join(args.epoch_output_dir, 'epoch_output')

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) > 0:
            args.model_name_or_path = sorted_checkpoints[-1]
        '''
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
        '''

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocabs

    if args.model_name_or_path:
        config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = GPT2Config()

    if args.model_name_or_path:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(GPT2Tokenizer.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = MAGICLMHeadModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = MAGICLMHeadModel(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, data_split='train')

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = MAGICLMHeadModel.from_pretrained(args.output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = MAGICLMHeadModel.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, data_split='test')
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    results = {}
    if args.do_generate and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Generate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = MAGICLMHeadModel.from_pretrained(checkpoint)
            model.to(args.device)
            result = generate_cmts(args, model, tokenizer, prefix=prefix, data_split='test')
            #result = dict((k + "_{}".format(str(global_step)), v) for k, v in result.items())
            #results.update(result)

    return results


if __name__ == "__main__":
    main()
