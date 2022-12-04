import argparse
from utils.utils import load_data, quick_tokenize, evaluation, define_logger
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

def parse_hps():
    parser = argparse.ArgumentParser(description='xCAR')

    # Data Paths
    parser.add_argument('--data_dir', type=str, default='./data/final_data/data/', help='The dataset directory')
    parser.add_argument('--model_dir', type=str, default='../../huggingface_transformers/xlnet-base-cased/',
                        help='The pretrained model directory')
    parser.add_argument('--save_dir', type=str, default='./output/saved_model', help='The model saving directory')
    parser.add_argument('--log_dir', type=str, default='./output/log', help='The training log directory')
    parser.add_argument('--apex_dir', type=str, default='./output/log', help='The apex directory')
    parser.add_argument('--output_dir', type=str, default='./')

    # Data names
    parser.add_argument('--train', type=str, default='train.pkl', help='The train data directory')
    parser.add_argument('--dev', type=str, default='dev.pkl', help='The dev data directory')
    parser.add_argument('--test', type=str, default='test.pkl', help='The test data directory')

    # Model Settings
    parser.add_argument('--model_name', type=str, default='xlnet', help='Pretrained model name')
    parser.add_argument('--save_name', type=str, default=None, help='Experiment save name')
    parser.add_argument('--data_name', type=str, default='copa')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu ids for training')
    # parser.add_argument('--apex', type=bool, default=False, help='Whether to use half precision')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=200, help='training iterations')
    parser.add_argument('--evaluation_strategy', type=str, default="step", help="evaluation metric [step] [epoch]")
    parser.add_argument('--evaluation_step', type=int, default=20,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1024, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=10, help='the patient of early-stopping')
    parser.add_argument('--loss_func', type=str, default='BCE', help="loss function of output")
    parser.add_argument('--hyp_only', type=bool, default=False, help="If set True, Only send hypothesis into model")
    parser.add_argument('--prompt', type=str, default=None, help="prompt template")
    # parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup settings')

    # parsing the hyper-parameters from command line and define logger
    hps = parser.parse_args()
    hps.nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return hps

def evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps, epoch, metric_log):
    model.eval()
    stop_train = False

    with torch.no_grad():
        print('\n')
        logger.info("[Dev Evaluation] Strain Evaluation on Dev Set")
        if hps.loss_func == 'CrossEntropy':
            dev_accu, dev_exact_accu, dev_loss = evaluation(hps, dev_dataloader, model, loss_function, epoch)
            evaluation_output = evaluation(hps, dev_dataloader, model, loss_function, epoch)
            metric_log[f'epoch_{epoch}'].update(evaluation_output)
            print('\n')
            logger.info("[Dev Metrics] Dev Soft Accuracy: \t{}".format(dev_accu))
            logger.info("[Dev Metrics] Dev Exact Accuracy: \t{}".format(dev_exact_accu))
        else:
            dev_accu, dev_loss = evaluation(hps, dev_dataloader, model, loss_function, epoch)
            evaluation_output = evaluation(hps, dev_dataloader, model, loss_function, epoch)
            metric_log[f'epoch_{epoch}'].update(evaluation_output)
            print('\n')
            logger.info("[Dev Metrics] Dev Accuracy: \t{}".format(dev_accu))
        logger.info("[Dev Metrics] Dev Loss: \t{}".format(dev_loss))

        with open(hps.output_dir + '/bert_cr_metric_log.json', 'w', encoding='utf-8') as fp:
            json.dump(metric_log, fp)

        if dev_accu >= best_accuracy:
            patient = 0
            best_accuracy = dev_accu
            if not os.path.exists(hps.save_dir):
                os.mkdir(hps.save_dir)
            logger.info("[Saving] Saving Model to {}".format(hps.save_dir))
            if hps.hyp_only:
                torch.save(model, os.path.join(hps.save_dir, 'discriminate_'+ hps.model_name + '_hyp'))
            else:
                torch.save(model, os.path.join(hps.save_dir, 'discriminate_'+ hps.model_name))
            # torch.save(model, os.path.join(hps.save_dir, exp_name))
                
        else:
            patient += 1

        logger.info("[Patient] {}".format(patient))
        if patient >= hps.patient:
            logger.info("[INFO] Stopping Training by Early Stopping")
            stop_train = True
    return patient, stop_train

def train(model, optimizer, train_dataloader, dev_dataloader, loss_function, logger, hps):
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
        for i, batch in zip(t, train_dataloader):
            optimizer.zero_grad()
            model.train()
            if hps.cuda:
                batch = tuple(term.cuda() for term in batch)

            sent, seg_id, atten_mask, labels, length = batch
            probs = model(sent, atten_mask, seg_ids=seg_id, length=length)

            if hps.loss_func == 'CrossEntropy':
                loss = loss_function(probs, labels)
            else:
                loss = loss_function(probs.squeeze(1), labels.float())

            total_loss += loss.item()
            t.set_postfix(avg_loss='{}'.format(total_loss/(epoch_step+1)))
            epoch_step += 1

            loss.backward()
            optimizer.step()

            if hps.evaluation_strategy == "step" and step % hps.evaluation_step == 0 and step != 0:
                patient, stop_train = evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps, epoch, metric_log)
                if stop_train:
                    return
            step += 1
        
        train_loss = total_loss / (epoch_step * hps.batch_size) * 100
        metric_log[f'epoch_{epoch}']['train_loss'] = train_loss
             
        if hps.evaluation_strategy == "epoch":
            patient, stop_train = evaluate(model, dev_dataloader, patient, best_accuracy, loss_function, logger, hps, epoch, metric_log)
            if stop_train:
                return

def load_loss_function(hps):
    if hps.loss_func == "CrossEntropy":
        loss_function = nn.CrossEntropyLoss(reduction='mean')
    elif hps.loss_func == "BCE":
        loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    return loss_function

def main():
    # parse hyper parameters
    hps = parse_hps()
    exp_name = "discriminate_" + hps.model_name
    if hps.save_name is not None:
        exp_name = hps.save_name + "_" + exp_name
    if hps.hyp_only:
        exp_name = exp_name + "_hyp"

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

    # tokenization
    logger.info("[DATA] Tokenization and Padding for Data")
    train_ids, train_mask, train_seg_ids, train_labels, train_length = quick_tokenize(train_data, hps)
    dev_ids, dev_mask, dev_seg_ids, dev_labels, dev_length = quick_tokenize(dev_data, hps)
    # test_ids, test_mask, test_seg_ids, test_labels, test_length = quick_tokenize(test_data, hps)
    print("tokenzied data:")
    print("\tdev_ids:", dev_ids[0])
    print("\tdev_mask:", dev_mask[0])
    print("\tdev_seg_ids:", dev_seg_ids[0])
    print("\tdev_labels:", dev_labels[0])
    print("\tdev_length:", dev_length[0])

    # Dataset and DataLoader
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
    model = pretrained_model(hps)
    logger.info(f"=== model architecture ===\n{model}")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
    loss_function = load_loss_function(hps)


    # multi-Gpu training
    if hps.cuda:
        gpu_ids = [int(x) for x in hps.gpu.split(',')]
        model = model.cuda()
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=gpu_ids)

    # training
    train(model, optimizer, train_dataloader, dev_dataloader, loss_function, logger, hps)

if __name__ == '__main__':
    main()





