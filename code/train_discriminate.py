import argparse
from utils.utils import parse_hps, get_exp_name, get_exp_path, load_data, quick_tokenize, load_loss_function, \
    cr_evaluation, define_logger, save_model
import random
import numpy as np
import torch
from model.discriminate_model import pretrained_model
# from transformers import AdamW
import sys
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import datetime
import logging
from collections import defaultdict
import json
from utils.kb import get_all_features
from utils.kb_dataset import MyDataLoader


def evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps, eval_step, metric_log, exp_name, exp_path):
    model.eval()
    stop_train = False

    with torch.no_grad():
        print('\n')
        logger.info("[Dev Evaluation] Start Evaluation on Dev Set")
        if hps.loss_func == 'CrossEntropy':
            dev_accu, dev_exact_accu, dev_loss = cr_evaluation(hps, dev_dataloader, model, loss_function, eval_step, exp_path)
            print('\n')
            logger.info("[Dev Metrics] Dev Soft Accuracy: \t{}".format(dev_accu))
            logger.info("[Dev Metrics] Dev Exact Accuracy: \t{}".format(dev_exact_accu))
        else:
            dev_accu, dev_loss = cr_evaluation(hps, dev_dataloader, model, loss_function, eval_step, exp_path)
            print('\n')
            logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(dev_accu))
        logger.info("[Dev Metrics] Dev Loss: \t{}".format(dev_loss))

        with open(exp_path + '/bert_cr_metric_log.json', 'w', encoding='utf-8') as fp:
            json.dump(metric_log, fp)

        if dev_accu >= best_accuracy:
            patient = 0
            best_accuracy = dev_accu
            save_model(model, hps, exp_name)
            logger.info("[Saving] Saving Model to {}".format(exp_path))

        else:
            patient += 1

        logger.info("[Patient] {}".format(patient))
        if patient >= hps.patient:
            logger.info("[INFO] Stopping Training by Early Stopping")
            stop_train = True
    return patient, stop_train, dev_accu, dev_loss


def train(model, optimizer, train_dataloader, dev_dataloader, loss_function, logger, hps, exp_name, exp_path):
    logger.info("[INFO] Start Training")
    step = 0
    patient = 0
    best_accuracy = 0
    stop_train = False
    metric_log = defaultdict(dict)

    for epoch in range(hps.epochs):
        logger.info('[Epoch] {}'.format(epoch))
        t = trange(len(train_dataloader))
        epoch_step = 0
        total_loss = 0
        for _, batch in zip(t, train_dataloader):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                device = f"cuda:{hps.gpu}"
                batch = tuple(term.to(device) for term in batch)
            if not hps.with_kb:
                sent, seg_id, attention_mask, labels, length = batch
                probs = model(sent, attention_mask, seg_ids=seg_id, length=length)
            else:
                sent, seg_id, attention_mask, labels, pos_ids = batch
                probs = model(sent, attention_mask, seg_ids=seg_id, position_ids=pos_ids)

            if hps.loss_func == 'CrossEntropy':
                loss = loss_function(probs, labels)
            elif hps.loss_func == "BCE":
                loss = loss_function(probs.squeeze(1), labels.float())

            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss / (epoch_step + 1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if hps.evaluation_strategy == "step" and step % hps.evaluation_step == 0 and step != 0:
                patient, stop_train, dev_accu, dev_loss = evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger,
                                               hps, step, metric_log, exp_name, exp_path)
                metric_log[f'step_{step}']['dev_accu'] = dev_accu
                metric_log[f'step_{step}']['dev_loss'] = dev_loss
                if stop_train:
                    return
            step += 1

        train_accu, train_loss = cr_evaluation(hps, train_dataloader, model, loss_function, eval_step, exp_path, print_pred=False)
        logger.info("[Train Metrics] Train Accuracy: \t{}".format(train_accu))
        logger.info("[Train Metrics] Train Loss: \t{}".format(train_loss))
        metric_log[f'epoch_{epoch}']['train_accu'] = train_accu
        metric_log[f'epoch_{epoch}']['train_loss'] = train_loss

        if hps.evaluation_strategy == "epoch":
            patient, stop_train, dev_accu, dev_loss = evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps,
                                           epoch, metric_log, exp_name, exp_path)
            metric_log[f'epoch_{epoch}']['dev_accu'] = dev_accu
            metric_log[f'epoch_{epoch}']['dev_loss'] = dev_loss
        if stop_train:
            return


def main():
    # parse hyper parameters
    hps = parse_hps()
    exp_name = get_exp_name(hps, "discriminate")
    exp_path = get_exp_path(hps, exp_name)

    # fix random seed
    if hps.set_seed:
        random.seed(hps.seed)
        np.random.seed(hps.seed)
        torch.manual_seed(hps.seed)
        torch.cuda.manual_seed(hps.seed)

    # prepare logger
    logger, formatter = define_logger()
    log_path = os.path.join(exp_path, exp_name + ".txt")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging all the hyper parameters
    logger.info(f"=== hps ===\n{hps}")

    logger.info(f"[INFO] Experiment Path: {exp_path}")

    # load data
    logger.info("[DATA] Loading Data")
    logger.info("[DATA] Hypothesis Only: {}".format(hps.hyp_only))
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    # test_data = load_data(os.path.join(hps.data_dir, hps.test))
    print("loaded data:", dev_data[0])

    # tokenization and data loading
    if not hps.with_kb:
        logger.info("[DATA] Tokenization and Padding for Data")
        train_ids, train_mask, train_seg_ids, train_labels, train_length = quick_tokenize(train_data, hps)
        dev_ids, dev_mask, dev_seg_ids, dev_labels, dev_length = quick_tokenize(dev_data, hps)
        # Dataset and DataLoader
        logger.info("[INFO] Creating Dataset and splitting batch for data")
        TRAIN = TensorDataset(train_ids, train_seg_ids, train_mask, train_labels, train_length)
        DEV = TensorDataset(dev_ids, dev_seg_ids, dev_mask, dev_labels, dev_length)
        train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
        dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    else:
        logger.info("[DATA] Tokenization and Padding for Data")
        train_ids, train_mask, train_seg_ids, train_pos_ids, train_labels = get_all_features(train_data, hps)
        dev_ids, dev_mask, dev_seg_ids, dev_pos_ids, dev_labels = get_all_features(dev_data, hps)
        # Dataset and DataLoader
        logger.info("[INFO] Creating Dataset and splitting batch for data")
        TRAIN = TensorDataset(train_ids, train_seg_ids, train_mask, train_labels, train_pos_ids)
        DEV = TensorDataset(dev_ids, dev_seg_ids, dev_mask, dev_labels, dev_pos_ids)
        train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
        dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')
    logger.info("[MODEL] {}".format(hps.model_name))
    model = pretrained_model(hps)
    # logger.info(f"=== model architecture ===\n{model}")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    loss_function = load_loss_function(hps)

    # multi-Gpu training
    if hps.cuda:
        gpu_ids = [int(x) for x in hps.gpu.split(',')]
        device = f"cuda:{hps.gpu}"
        model = model.to(device)
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=gpu_ids)

    # training
    train(model, optimizer, train_dataloader, dev_dataloader, loss_function, logger, hps, exp_name, exp_path)


if __name__ == '__main__':
    main()