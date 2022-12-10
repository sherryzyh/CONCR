import argparse
from utils.utils import parse_hps, get_exp_name, get_exp_path, load_data, define_logger, tokenize_gen, gpt2_eg_evaluate, compute_ppl, save_metric_log, save_model
import random
import numpy as np
import torch
# from model.generatively_model import gpt2_generate, bart_generate
from model.generatively_model import gpt2_generate
from transformers import AdamW, GPT2LMHeadModel
import sys
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import datetime
import logging
import pdb
from collections import defaultdict
import json

def train(model, optimizer, train_dataloader, dev_dataloader, dev_data, logger, hps, exp_name, exp_path):
    # training
    logger.info("[INFO] Start Training")
    patient = 0
    best_loss = 0
    stop_train = False
    metric_log = defaultdict(dict)

    for epoch in range(hps.epochs):
        logger.info('[Epoch] {}'.format(epoch))
        t = trange(len(train_dataloader))
        epoch_step = 0
        train_loss = 0
        for i, batch in zip(t, train_dataloader):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                device = f"cuda:{hps.gpu}"
                batch = tuple(term.to(device) for term in batch)

            input_ids, input_mask, input_seg_ids, input_labels, input_labels_mask = batch
            tmp = torch.ones(input_labels_mask.shape).long()
            count_mask_length = torch.sum(tmp==input_labels_mask.cpu(), 1).squeeze().tolist()
            true_labels = None
            for j in range(input_ids.shape[0]):
                if true_labels is None:
                    # true_labels = torch.cat((torch.ones(count_mask_length[j]).long(), input_ids[j, count_mask_length[j]:].cpu())).unsqueeze(0)
                    true_labels = torch.cat((input_ids[j, :-count_mask_length[j]]*0-100, input_ids[j, -count_mask_length[j]:])).unsqueeze(0)
                else:
                    # true_labels = torch.cat((true_labels, torch.cat((torch.ones(count_mask_length[j]).long(), input_ids[j, count_mask_length[j]:].cpu())).unsqueeze(0)), 0)
                    true_labels = torch.cat((true_labels, torch.cat((input_ids[j, :-count_mask_length[j]]*0-100, input_ids[j, -count_mask_length[j]:])).unsqueeze(0)),0)

            output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg_ids, labels=true_labels)
            loss = output[0]

            train_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(train_loss / (epoch_step+1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()
        train_loss = train_loss / (epoch_step * hps.batch_size) * 100
        metric_log[f'epoch_{epoch}']['train_loss'] = train_loss

        model.eval()

        # by default, evaluation strategy "epoch"
        print('\n')
        logger.info("[Dev Evaluation] Start Evaluation on Dev Set")
        evaluation_output = gpt2_eg_evaluate(hps, dev_dataloader, model, epoch, exp_path)
        dev_ppl = compute_ppl(hps, model, dev_data)

        metric_log[f'epoch_{epoch}'].update(evaluation_output)
        metric_log[f'epoch_{epoch}']['perplexity'] = dev_ppl

        logger.info("[Train Metrics] Train Loss: \t{}".format(train_loss))
        logger.info("[Dev Metrics] Dev Loss: \t{}".format(evaluation_output['val_loss']))
        logger.info("[Dev Metrics] Average BLEU:\t{}".format(evaluation_output['avg_bleu']))
        logger.info("[Dev Metrics] Rouge:\t({}, {}, {})".format(
            evaluation_output['rouge1'], evaluation_output['rouge2'], evaluation_output['rougel']))
        logger.info("[Dev Metrics] Perplexity: \t{}".format(dev_ppl))

        save_metric_log(metric_log, hps, exp_name)

        # save best model
        if epoch == 0 or evaluation_output['val_loss'] < best_loss:
            best_loss = evaluation_output['val_loss']
            save_model(model, hps, exp_name, mode="minloss")
            logger.info("[Saving] Saving Model to {}".format(hps.save_dir))


def main():
    # parse hyper parameters
    hps = parse_hps()
    exp_name = get_exp_name(hps, "generate")
    exp_path = get_exp_path(hps, exp_name)

    # fix random seed
    if hps.set_seed:
        random.seed(hps.seed)
        np.random.seed(hps.seed)
        torch.manual_seed(hps.seed)
        torch.cuda.manual_seed(hps.seed)


    # prepare logger
    logger, formatter = define_logger()
    log_path = os.path.join(hps.log_dir, 'prompt1_generated_'+hps.model_name+'.txt')
    log_path = os.path.join(exp_path, exp_name + ".txt")

    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging all the hyper parameters
    logger.info(f"=== hps ===\n{hps}")

    logger.info(f"[INFO] Experiment Path: {exp_path}")

    # load data
    # logger.info("[Pytorch] %s", torch.)
    logger.info("[INFO] Loading Data")
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    # test_data = load_data(os.path.join(hps.data_dir, hps.test))

    # Tokenization
    logger.info("[INFO] Tokenization and Padding for Data")
    train_ids, train_mask, train_seg_ids, train_label_ids, train_label_mask, _, _, _, _ = tokenize_gen(train_data, hps)
    dev_ids, dev_mask, dev_seg_ids, dev_label_ids, dev_label_mask, dev_label_seg_ids, dev_premise_ids, dev_premise_mask, dev_premise_seg_ids = tokenize_gen(dev_data, hps)
    # _, _, _, test_label_ids, test_label_mask, test_label_seg_ids, test_premise_ids, test_premise_mask, test_premise_seg_ids = tokenize_gen(test_data, hps)

    # Dataset and DataLoader
    logger.info("[INFO] Creating Dataset and splitting batch for data")
    TRAIN = TensorDataset(train_ids, train_mask, train_seg_ids, train_label_ids, train_label_mask)
    DEV = TensorDataset(dev_ids, dev_mask, dev_seg_ids, dev_label_ids, dev_label_mask, dev_label_seg_ids, dev_premise_ids, dev_premise_mask, dev_premise_seg_ids)
    # TEST = TensorDataset(test_label_ids, test_label_mask, test_label_seg_ids, test_premise_ids, test_premise_mask, test_premise_seg_ids)
    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    # test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')

    # model = gpt2_generate(hps)
    model = GPT2LMHeadModel.from_pretrained(hps.model_dir, pad_token_id=50256)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    # Multi-Gpu training
    if hps.cuda:
        gpu_ids = [int(x) for x in hps.gpu.split(',')]
        device = f"cuda:{hps.gpu}"
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

    train(model, optimizer, train_dataloader, dev_dataloader, dev_data, logger, hps, exp_name, exp_path)

if __name__ == '__main__':
    main()
