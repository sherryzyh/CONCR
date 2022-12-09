import argparse
from utils.utils import parse_hps, get_exp_name, load_data, quick_tokenize, contrastive_tokenize, load_loss_function, evaluation, cl_evaluation, define_logger
import random
import numpy as np
import torch
from model.discriminate_model import pretrained_model
from model.contrastive_discriminate_model import contrastive_reasoning_model
# from transformers import AdamW
import sys
import torch.nn as nn
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import datetime
import logging


def CL_evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps, exp_name, mode="dev"):
    model.eval()
    stop_train = False

    with torch.no_grad():
        print('\n')
        logger.info("[Dev Evaluation] Start Evaluation on Dev Set")
        if hps.loss_func == 'CrossEntropy':
            dev_accu, dev_exact_accu, dev_loss = cl_evaluation(hps, dev_dataloader, model, loss_function)
            print('\n')
            logger.info("[Dev Metrics] Dev Soft Accuracy: \t{}".format(dev_accu))
            logger.info("[Dev Metrics] Dev Exact Accuracy: \t{}".format(dev_exact_accu))
        else:
            dev_accu, dev_loss = cl_evaluation(hps, dev_dataloader, model, loss_function)
            print('\n')
            logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(dev_accu))
        logger.info("[Dev Metrics] Dev Loss: \t{}".format(dev_loss))

        if dev_accu >= best_accuracy:
            patient = 0
            best_accuracy = dev_accu
            if not os.path.exists(hps.save_dir):
                os.mkdir(hps.save_dir)
            logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
            torch.save(model, os.path.join(hps.save_dir, exp_name))

        else:
            patient += 1

        logger.info("[Patient] {}".format(patient))
        if patient >= hps.patient:
            logger.info("[INFO] Stopping Training by Early Stopping")
            stop_train = True
    return patient, stop_train


def CL_train(model, optimizer, train_dataloader, dev_dataloader, loss_function, logger, hps, exp_name):
    logger.info("[INFO] Start Training")
    step = 0
    patient = 0
    best_accuracy = 0
    stop_train = False
    for epoch in range(hps.epochs):
        logger.info('[Epoch] {}'.format(epoch))
        t = trange(len(train_dataloader))
        epoch_step = 0
        total_loss = 0
        for i, batch in zip(t, train_dataloader):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                batch = tuple(term.cuda() for term in batch)

            sent, seg_id, atten_mask, labels, length = batch
            output = model.forward(sent, atten_mask, labels, seg_ids=seg_id, length=length, mode='train')
            loss = output.loss

            total_loss += loss.item()
            if i == 0:
                init_loss = loss.item() / len(batch)
            last_loss = loss.item() / len(batch)
            t.set_postfix(avg_loss='{}'.format(total_loss / (epoch_step + 1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if hps.evaluation_strategy == "step" and step % hps.evaluation_step == 0 and step != 0:
                patient, stop_train = CL_evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger,
                                               hps, exp_name)
                if stop_train:
                    return
            step += 1

        if hps.loss_func == 'BCE':
            train_accu, train_loss = cl_evaluation(hps, train_dataloader, model, loss_function)
        logger.info("[Train Metrics] Train Accuracy: \t{}".format(train_accu))
        logger.info("[Train Metrics] Train Loss: \t{}".format(train_loss))

        if hps.evaluation_strategy == "epoch":
            patient, stop_train = CL_evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps,
                                           exp_name)
        if stop_train:
            return

def main():
    # parse hyper parameters
    hps = parse_hps()
    exp_name = get_exp_name(hps, "discriminate")

    # fix random seed
    if hps.set_seed:
        random.seed(hps.seed)
        np.random.seed(hps.seed)
        torch.manual_seed(hps.seed)
        torch.cuda.manual_seed(hps.seed)

    # prepare logger
    logger, formatter = define_logger()
    log_path = os.path.join(hps.log_dir, exp_name + ".txt")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging all the hyper parameters
    logger.info(f"=== hps ===\n{hps}")

    # load data
    logger.info("[DATA] Loading Data")
    logger.info("[DATA] Hypothesis Only: {}".format(hps.hyp_only))
    train_data = load_data(os.path.join(hps.data_dir, hps.train))
    dev_data = load_data(os.path.join(hps.data_dir, hps.dev))
    # test_data = load_data(os.path.join(hps.data_dir, hps.test))
    print("loaded data:", dev_data[0])

    # contrastive Tokenization
    logger.info("[DATA] Tokenization and Padding for Data")
    train_ids, train_mask, train_seg_ids, train_labels, train_length = contrastive_tokenize(train_data, hps,
                                                                                            loading_mode="train")
    dev_ids, dev_mask, dev_seg_ids, dev_labels, dev_length = contrastive_tokenize(dev_data, hps,
                                                                                  loading_mode=hps.devload)

    # contrastive Dataset and DataLoader
    logger.info("[INFO] Creating Dataset and splitting batch for data")
    TRAIN = TensorDataset(train_ids, train_seg_ids, train_mask, train_labels, train_length)
    DEV = TensorDataset(dev_ids, dev_seg_ids, dev_mask, dev_labels, dev_length)
    # TEST = TensorDataset(test_ids, test_seg_ids, test_mask, test_labels, test_length)
    train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
    # test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

    # initialize model, optimizer, loss_function
    logger.info('[INFO] Loading pretrained model, setting optimizer and loss function')
    logger.info("[MODEL] {}".format(hps.model_name))
    model = contrastive_reasoning_model(hps)

    # logger.info(f"=== model architecture ===\n{model}")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    loss_function = load_loss_function(hps)

    # multi-Gpu training
    if hps.cuda:
        gpu_ids = [int(x) for x in hps.gpu.split(',')]
        model = model.cuda()
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=gpu_ids)

    # contrastive training
    CL_train(model, optimizer, train_dataloader, dev_dataloader, loss_function, logger, hps, exp_name)


if __name__ == '__main__':
    main()