import argparse
from utils.utils import parse_hps, get_exp_name, get_exp_path, load_data, quick_tokenize, dual_tokenize, \
    load_loss_function, vanilla_cr_evaluation, siamese_cr_evaluation, define_logger, save_model, save_metric_log
import random
import numpy as np
import torch
from model.discriminate_model import pretrained_model
from model.siamese_discriminate_model import siamese_reasoning_model
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
import spacy


def evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps, eval_step, metric_log, exp_name,
             exp_path):
    model.eval()
    stop_train = False

    with torch.no_grad():
        print('\n')
        logger.info("[Dev Evaluation] Start Evaluation on Dev Set")

        if hps.model_architecture == "siamese":
            dev_accu, dev_loss = siamese_cr_evaluation(hps, dev_dataloader, model, loss_function, eval_step, exp_path)
        else:
            dev_accu, dev_loss = vanilla_cr_evaluation(hps, dev_dataloader, model, loss_function, eval_step, exp_path)
        print('\n')
        logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(dev_accu))
        logger.info("[Dev Metrics] Dev Loss: \t{}".format(dev_loss))

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

            if hps.with_kb:
                sent, seg_id, attention_mask, labels, pos_ids = batch
                probs = model(sent, attention_mask, seg_ids=seg_id, position_ids=pos_ids)
            elif hps.model_architecture == "siamese":
                sent, seg_id, attention_mask, labels, length, ask_for = batch
                probs = model(sent, attention_mask, ask_for=ask_for, seg_ids=seg_id, length=length)
            else:
                sent, seg_id, attention_mask, labels, length = batch
                probs = model(sent, attention_mask, seg_ids=seg_id, length=length)

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
                patient, stop_train, dev_accu, dev_loss = evaluate(model, dev_dataloader, patient, best_accuracy,
                                                                   loss_function, logger,
                                                                   hps, step, metric_log, exp_name, exp_path)
                metric_log[f'step_{step}']['dev_accu'] = dev_accu
                metric_log[f'step_{step}']['dev_loss'] = dev_loss
                if stop_train:
                    return
            step += 1

        if hps.model_architecture == "siamese":
            train_accu, train_loss = siamese_cr_evaluation(hps, train_dataloader, model, loss_function, epoch, exp_path,
                                                           print_pred=False)
        else:
            train_accu, train_loss = vanilla_cr_evaluation(hps, train_dataloader, model, loss_function, epoch, exp_path,
                                                           print_pred=False)

        logger.info("[Train Metrics] Train Accuracy: \t{}".format(train_accu))
        logger.info("[Train Metrics] Train Loss: \t{}".format(train_loss))
        metric_log[f'epoch_{epoch}']['train_accu'] = train_accu
        metric_log[f'epoch_{epoch}']['train_loss'] = train_loss

        if hps.evaluation_strategy == "epoch":
            patient, stop_train, dev_accu, dev_loss = evaluate(model, dev_dataloader, patient, best_accuracy,
                                                               loss_function, logger, hps,
                                                               epoch, metric_log, exp_name, exp_path)
            metric_log[f'epoch_{epoch}']['dev_accu'] = dev_accu
            metric_log[f'epoch_{epoch}']['dev_loss'] = dev_loss

        save_metric_log(metric_log, hps, exp_name)

        if stop_train:
            return


def load_tokenized_dataset(hps, data, mode, nlp=None):
    if hps.with_kb:
        data_ids, data_mask, data_seg_ids, data_pos_ids, data_labels = get_all_features(data, hps, nlp)
        DATA = TensorDataset(data_ids, data_mask, data_seg_ids, data_pos_ids, data_labels)
    elif hps.model_architecture == "siamese":
        data_ids, data_mask, dataseg_ids, data_labels, data_length, data_ask_for = dual_tokenize(data, hps, mode)
        DATA = TensorDataset(data_ids, data_mask, dataseg_ids, data_labels, data_length, data_ask_for)
    else:  # standard
        data_ids, data_mask, dataseg_ids, data_labels, data_length = quick_tokenize(data, hps)
        DATA = TensorDataset(data_ids, data_mask, dataseg_ids, data_labels, data_length)

    return DATA


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

    logger.info(f"[INFO] Experiment Path: {exp_path}")

    # logging all the hyper parameters
    logger.info(f"=== hps ===\n{hps}")

    # load data
    logger.info("[DATA] Loading Data")
    logger.info("[DATA] Hypothesis Only: {}".format(hps.hyp_only))
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    # test_data = load_data(os.path.join(hps.data_dir, hps.test))

    # tokenization and data loading
    logger.info("[DATA] Tokenization and Padding for Data")
    if hps.with_kb:
        nlp = spacy.load('en_core_web_sm')
    else:
        nlp = None
    TRAIN = load_tokenized_dataset(hps, train_data, "train", nlp)
    print("train data tokenized")
    DEV = load_tokenized_dataset(hps, dev_data, "dev", nlp)
    print("dev data tokenized")

    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')
    logger.info("[MODEL] {}".format(hps.model_name))

    if hps.model_architecture == "single":
        model = pretrained_model(hps)
    elif hps.model_architecture == "siamese":
        model = siamese_reasoning_model(hps)
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
