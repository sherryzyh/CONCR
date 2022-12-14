import pickle
import argparse
import os
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, OpenAIGPTTokenizer, XLNetTokenizer
from transformers import GPT2Tokenizer, BartTokenizer
import torch
import logging
import sys
from rouge import Rouge
from nltk import bleu
from tqdm import trange
import torch.nn.functional as F
from tqdm import trange
from nlp import load_dataset
from tqdm import tqdm
import datetime
import csv
import pdb
import torch.nn as nn
import json
import pandas as pd
import nltk.stem as ns
from .kb_cl import get_all_x_features

"""
    Helper
"""

ALPHA = 0.66
LAMBDA = 1


def parse_hps():
    parser = argparse.ArgumentParser(description='ECR-ANLP')

    # Data Paths
    parser.add_argument('--data_dir', type=str, default='data/final_data/data/', help='The dataset directory')
    parser.add_argument('--model_dir', type=str, default='huggingface_cache/xlnet-base-cased/',
                        help='The pretrained model directory')
    parser.add_argument('--save_dir', type=str, default='output/saved_model', help='The model saving directory')
    parser.add_argument('--log_dir', type=str, default='output/log', help='The training log directory')
    parser.add_argument('--apex_dir', type=str, default='output/log', help='The apex directory')
    parser.add_argument('--storage', type=bool, default=False, help='Whether to use external storage')
    parser.add_argument('--storage_dir', type=str, default='/data', help='The storage root directory')

    # Data names
    parser.add_argument('--train', type=str, default='train.pkl', help='The train data directory')
    parser.add_argument('--dev', type=str, default='dev.pkl', help='The dev data directory')
    parser.add_argument('--devload', type=str, default='train', help='loading mode for dev dataset')
    parser.add_argument('--test', type=str, default='test.pkl', help='The test data directory')

    # Model Settings
    parser.add_argument('--model_architecture', type=str, default="single",
                        help='Model Architecture. Options: [single][siamese]')
    parser.add_argument('--model_name', type=str, default='xlnet', help='Pretrained model name')
    parser.add_argument('--save_name', type=str, default=None, help='Experiment save name')
    parser.add_argument('--data_name', type=str, default='copa')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu ids for training')
    parser.add_argument('--apex', type=bool, default=False, help='Whether to use half precision')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size for training and evaluation')
    parser.add_argument('--shuffle', type=bool, default=False, help='whether to shuffle training data')
    parser.add_argument('--epochs', type=int, default=200, help='training iterations')
    parser.add_argument('--evaluation_strategy', type=str, default="step", help="evaluation metric [step] [epoch]")
    parser.add_argument('--evaluation_step', type=int, default=20,
                        help='when training for some steps, start evaluation')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate of training')
    parser.add_argument('--use_wd', type=bool, default=False, help='Whether to add weight decay')
    parser.add_argument('--wd', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix the random seed')
    parser.add_argument('--seed', type=int, default=1024, help='fix the random seed for reproducible')
    parser.add_argument('--patient', type=int, default=10, help='the patient of early-stopping')
    parser.add_argument('--loss_func', type=str, default='BCE', help="loss function of output")
    parser.add_argument('--hyp_only', type=bool, default=False, help="If set True, Only send hypothesis into model")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup settings')

    # Method Settings
    parser.add_argument('--with_kb', type=str, default=False, help='Whether to use knowledge base')
    parser.add_argument('--with_cl', type=str, default=False, help='Whether to use knowledge base')
    parser.add_argument('--dual_projecter', type=bool, default=False, help='Whether to use cause/effect projecter')
    parser.add_argument('--projecterlr', type=float, default=1e-3, help="the learning rate of the projecter")
    parser.add_argument('--score', type=str, default="cossim", help="scorer type")
    parser.add_argument('--scorerlr', type=float, default=1e-3, help="the learning rate of the causal scorer")
    parser.add_argument('--hard_negative_weight', type=float, default=0.0, help="hard negative weight")
    parser.add_argument('--prompt', type=str, default=None, help="prompt template, options: [T0][T1][T2] etc.")
    parser.add_argument('--length', type=int, default=22, help='the max length of generated text')

    # parsing the hyper-parameters from command line and define logger
    hps = parser.parse_args()
    hps.nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return hps


def get_exp_name(hps, task):
    exp_name = task + "_" + hps.model_dir.split("/")[-1]
    if hps.hyp_only:
        exp_name = exp_name + "_hyp"
    if hps.save_name is not None:
        exp_name = hps.save_name + "_" + exp_name
    if hps.with_kb:
        exp_name = "kb_" + exp_name
    if hps.with_cl:
        exp_name = "cl_" + exp_name
    if task == "generate":
        if hps.prompt is None:
            exp_name = "vanilla_" + exp_name
        else:
            exp_name = hps.prompt + "_" + exp_name
    elif task == "discriminate":
        if hps.model_architecture != "single":
            exp_name = hps.model_architecture + "_" + exp_name

    return exp_name


def load_loss_function(hps):
    if hps.loss_func == "CrossEntropy":
        loss_function = nn.CrossEntropyLoss(reduction='mean')
    elif hps.loss_func == "BCE":
        loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    return loss_function


def load_data(path):
    data = [json.loads(line) for line in open(path, 'r')]
    return data


def print_prediction(exp_path, task, hps, eval_step, predictions):
    print_path = os.path.join(exp_path, "predictions")
    if not os.path.exists(print_path):
        os.mkdir(print_path)
    pred_path = os.path.join(print_path, f"{task}_pred_{hps.evaluation_strategy}_{eval_step}.csv")

    if "cr" in task:
        with open(pred_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows([[l] for l in predictions])
    elif "eg" in task:
        with open(pred_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(predictions)


def define_logger():
    logger = logging.getLogger('Discriminate logger')
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    return logger, formatter


def get_exp_path(hps, exp_name):
    if hps.storage:
        exp_path = os.path.join(hps.storage_dir, hps.save_dir, exp_name)
    else:
        exp_path = os.path.join(hps.save_dir, exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    return exp_path


def save_model(model, hps, exp_name, mode="best"):
    exp_path = get_exp_path(hps, exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    if mode == "best":
        torch.save(model, os.path.join(exp_path, "best_acc_ckpt.pt"))
    elif mode == "minloss":
        torch.save(model, os.path.join(exp_path, "best_loss_ckpt.pt"))


def save_metric_log(metric_log, hps, exp_name):
    exp_path = get_exp_path(hps, exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    with open(os.path.join(exp_path, "metric_log.json"), 'w', encoding='utf-8') as fp:
        json.dump(metric_log, fp)


"""
    Tokenization
"""


def load_pretrained_tokenizer(hps):
    # load pretrained tokenizer
    if hps.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(hps.model_dir, unk_token="<unk>", padding_side='left')
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir, padding_side='left')
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    else:
        tokenizer = XLNetTokenizer.from_pretrained(hps.model_dir, padding_side='left')

    return tokenizer


def tokenize_data(data, model_path, model_name):
    # tokenizer = BertTokenizer(vocab_file=model_path+'/'+'vocab.txt')
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_path, padding_side='left')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(model_path, padding_side='left')

    # unique ids
    cls_id = tokenizer._convert_token_to_id('[CLS]')
    sep_id = tokenizer._convert_token_to_id('[SEP]')
    pad_id = tokenizer._convert_token_to_id('[PAD]')

    labels = []
    instances = []
    segments = []

    max_length = 0

    # tokenization
    for example in data:
        premise, a1, a2 = example['premise'], example['hypothesis1'], example['hypothesis2']
        premise_id = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(premise))
        a1_id = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(a1))
        a2_id = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(a2))
        max_length = max(max_length, len(premise_id + a1_id) + 3, len(premise_id + a2_id) + 3)
        if example['ask-for'] == 'cause':
            instance1 = [cls_id] + a1_id + [sep_id] + premise_id + [sep_id]
            seg1 = [0] * (len(a1_id) + 2) + [1] * (len(premise_id) + 1)
            instance2 = [cls_id] + a2_id + [sep_id] + premise_id + [sep_id]
            seg2 = [0] * (len(a2_id) + 2) + [1] * (len(premise_id) + 1)
        else:
            instance1 = [cls_id] + premise_id + [sep_id] + a1_id + [sep_id]
            seg1 = [0] * (len(premise_id) + 2) + [1] * (len(a1_id) + 1)
            instance2 = [cls_id] + premise_id + [sep_id] + a2_id + [sep_id]
            seg2 = [0] * (len(premise_id) + 2) + [1] * (len(a2_id) + 1)
        instances += [instance1, instance2]
        segments += [seg1, seg2]
        labels += [0, 1] if example['label'] == 1 else [1, 0]

    # padding
    segments = [seg + [0] * (max_length - len(seg)) for seg in segments]
    attention_mask = [[1] * len(instance) + [0] * (max_length - len(instance)) for instance in instances]
    instances = [instance + [pad_id] * (max_length - len(instance)) for instance in instances]

    return torch.LongTensor(instances), torch.LongTensor(attention_mask), torch.LongTensor(segments), torch.LongTensor(
        labels)


def tokenize_multi_choices(data, hps):
    # load pretrained tokenizer
    if hps.model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'gpt':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(hps.model_dir, unk_token="<unk>", padding_side='left')
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir, padding_side='left')
        tokenizer.pad_token = tokenizer.unk_token
    elif hps.model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    else:
        tokenizer = XLNetTokenizer.from_pretrained(hps.model_dir, padding_side='left')

    instances = []
    labels = []

    # pdb.set_trace()
    for example in data:
        if hps.data_name == 'because' or hps.data_name == 'event_storyline':
            premise, hypothesis = example['premise'], example['hypothesis']
            instance = [premise, hypothesis]
            labels.append(example['label'])
            instances.append(instance)
        elif hps.data_name == 'commonsenseqa':
            premise, alternatives = example['premise'], example['alternatives']
            label = example['label']
            tmp_instances = [[premise, alternative] for alternative in alternatives]
            tmp_labels = [0 for _ in range(len(alternatives))]
            tmp_labels[label] = 1
            labels += tmp_labels
            instances += tmp_instances

    outputs = tokenizer(instances, padding=True, return_token_type_ids=True, return_length=True)
    input_ids = outputs['input_ids']
    attention_mask = outputs['attention_mask']
    token_type_ids = outputs['token_type_ids']
    length = outputs['length']

    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), \
           torch.LongTensor(token_type_ids), torch.LongTensor(labels), torch.LongTensor(length) - 1


def contrastive_tokenize(data, hps, loading_mode="train"):
    tokenizer = load_pretrained_tokenizer(hps)

    instances = []
    labels = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    length = []
    for i, example in enumerate(data):
        if loading_mode == "train":
            premise, hyp0, hyp1 = example['premise'], example['hypothesis1'], example['hypothesis2']
            if example['label'] == 0:
                instance = [premise, hyp0, hyp1]
            else:
                instance = [premise, hyp1, hyp0]
            instances += instance
            # labels: [['0' for 'ask-for-cause'/'1' for 'ask-for-effect',
            #           1 for correct hypothesis
            #           0 for wrong hypothesis] x n samples]
            labels += [0, 1, 0] if example['ask-for'] == 'cause' else [1, 1, 0]
        elif loading_mode == "dev":
            premise, hyp0, hyp1 = example['premise'], example['hypothesis1'], example['hypothesis2']
            instances += [premise, hyp0, hyp1]
            label = [0, 0, 0]
            label[example['label'] + 1] = 1
            label[0] = 0 if example['ask-for'] == 'cause' else 1
            labels += label

    tokenized_inputs = tokenizer(text=instances, padding=True, return_token_type_ids=True, return_length=True)
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']
    token_type_ids = tokenized_inputs['token_type_ids']
    length = tokenized_inputs['length']

    data_size = len(data)
    max_len = max(length)
    input_ids_tensor = torch.LongTensor(input_ids).view((data_size, 3, max_len))
    attention_mask_tensor = torch.LongTensor(attention_mask).view((data_size, 3, max_len))
    token_type_ids_tensor = torch.LongTensor(token_type_ids).view((data_size, 3, max_len))
    labels_tensor = torch.LongTensor(labels).view((data_size, 3))
    length_tensor = torch.LongTensor(length).view((data_size, 3)) - 1

    return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, labels_tensor, length_tensor


def contrastive_kb_tokenize(data, hps, nlp, loading_mode="train"):
    tokenizer = load_pretrained_tokenizer(hps)

    instances = []
    labels = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    length = []
    for i, example in enumerate(data):
        if loading_mode == "train":
            premise, hyp0, hyp1 = example['premise'], example['hypothesis1'], example['hypothesis2']
            if example['label'] == 0:
                instance = [premise, hyp0, hyp1]
            else:
                instance = [premise, hyp1, hyp0]
            instances += instance
            # labels: [['0' for 'ask-for-cause'/'1' for 'ask-for-effect',
            #           1 for correct hypothesis
            #           0 for wrong hypothesis] x n samples]
            labels += [0, 1, 0] if example['ask-for'] == 'cause' else [1, 1, 0]
        elif loading_mode == "dev":
            premise, hyp0, hyp1 = example['premise'], example['hypothesis1'], example['hypothesis2']
            instances += [premise, hyp0, hyp1]
            label = [0, 0, 0]
            label[example['label'] + 1] = 1
            label[0] = 0 if example['ask-for'] == 'cause' else 1
            labels += label

    # tokenized_inputs = tokenizer(text=instances, padding=True, return_token_type_ids=True, return_length=True)
    # input_ids = tokenized_inputs['input_ids']
    # attention_mask = tokenized_inputs['attention_mask']
    # token_type_ids = tokenized_inputs['token_type_ids']
    # length = tokenized_inputs['length']
    input_ids, attention_mask, token_type_ids, soft_pos_ids = get_all_x_features(tokenizer, instances, hps, nlp)

    data_size = len(data)
    max_len = 128
    input_ids_tensor = torch.LongTensor(input_ids).view((data_size, 3, max_len))
    attention_mask_tensor = torch.LongTensor(attention_mask).view((data_size, 3, max_len, max_len))
    token_type_ids_tensor = torch.LongTensor(token_type_ids).view((data_size, 3, max_len))
    soft_pos_ids_tensor = torch.LongTensor(soft_pos_ids).view((data_size, 3, max_len))
    labels_tensor = torch.LongTensor(labels).view((data_size, 3))

    return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, labels_tensor, soft_pos_ids_tensor


def dual_tokenize(data, hps, mode="train"):
    tokenizer = load_pretrained_tokenizer(hps)

    instances = []
    labels = []
    ask_for = []
    for example in data:
        premise, hyp0, hyp1 = example['premise'], example['hypothesis1'], example['hypothesis2']
        if mode == "train":
            instances += [premise, hyp0]
            instances += [premise, hyp1]
            labels += [0, 1] if example['label'] == 1 else [1, 0]
            ask_for += [0, 0] if example['ask-for'] == 'cause' else [1, 1]
        elif mode == "dev":
            premise, hyp0, hyp1 = example['premise'], example['hypothesis1'], example['hypothesis2']
            instances += [premise, hyp0, hyp1]
            label = [0, 1] if example['label'] == 1 else [1, 0]
            labels.append(label)
            ask_for += [0] if example['ask-for'] == 'cause' else [1]

    tokenized_inputs = tokenizer(instances, padding=True, return_token_type_ids=True, return_length=True)
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']
    token_type_ids = tokenized_inputs['token_type_ids']
    length = tokenized_inputs['length']

    data_size = len(data)
    max_len = max(length)

    if mode == "train":
        input_ids_tensor = torch.LongTensor(input_ids).view((data_size * 2, 2, max_len))
        attention_mask_tensor = torch.LongTensor(attention_mask).view((data_size * 2, 2, max_len))
        token_type_ids_tensor = torch.LongTensor(token_type_ids).view((data_size * 2, 2, max_len))
        length_tensor = torch.LongTensor(length).view((data_size * 2, 2)) - 1

    elif mode == "dev":
        input_ids_tensor = torch.LongTensor(input_ids).view((data_size, 3, max_len))
        attention_mask_tensor = torch.LongTensor(attention_mask).view((data_size, 3, max_len))
        token_type_ids_tensor = torch.LongTensor(token_type_ids).view((data_size, 3, max_len))
        length_tensor = torch.LongTensor(length).view((data_size, 3)) - 1

    labels_tensor = torch.LongTensor(labels)
    ask_for_tensor = torch.LongTensor(ask_for)

    return input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, labels_tensor, \
           length_tensor, ask_for_tensor


def quick_tokenize(data, hps):
    tokenizer = load_pretrained_tokenizer(hps)

    instances = []
    labels = []
    for example in data:
        premise, a1, a2 = example['premise'], example['hypothesis1'], example['hypothesis2']

        if example['ask-for'] == 'cause':
            if not hps.hyp_only:
                instance1 = [a1, premise]
                instance2 = [a2, premise]
            else:
                instance1 = a1
                instance2 = a2
        else:
            if not hps.hyp_only:
                instance1 = [premise, a1]
                instance2 = [premise, a2]
            else:
                instance1 = a1
                instance2 = a2
        labels += [0, 1] if example['label'] == 1 else [1, 0]
        instances += [instance1, instance2]

    outputs = tokenizer(instances, padding=True, return_token_type_ids=True, return_length=True)
    input_ids = outputs['input_ids']
    attention_mask = outputs['attention_mask']
    token_type_ids = outputs['token_type_ids']
    length = outputs['length']

    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), \
           torch.LongTensor(token_type_ids), torch.LongTensor(labels), torch.LongTensor(length) - 1


def tokenize_multi_task(hps, data):
    tokenizer = RobertaTokenizer.from_pretrained(hps.discriminate_model_dir, padding_side='left')
    instances1 = []
    instances2 = []
    labels = []
    truths = []

    for example in data:
        truth, premise, a1, a2 = example['conceptual_explanation'], example['premise'], example['hypothesis1'], example[
            'hypothesis2']
        truths.append(truth)
        if example['ask-for'] == 'cause':
            instances1.append([a1, premise])
            instances2.append([a2, premise])
        else:
            instances1.append([premise, a1])
            instances2.append([premise, a2])
        labels += [example['label']]

    outputs1 = tokenizer(instances1, padding=True)
    outputs2 = tokenizer(instances2, padding=True)
    outputs_truth = tokenizer(truths, padding=True)
    input_ids1 = torch.LongTensor(outputs1['input_ids'])
    input_ids2 = torch.LongTensor(outputs2['input_ids'])
    truth_ids = torch.LongTensor(outputs_truth['input_ids'])
    mask1 = torch.LongTensor(outputs1['attention_mask'])
    mask2 = torch.LongTensor(outputs2['attention_mask'])
    mask_truth = torch.LongTensor(outputs_truth['attention_mask'])

    return input_ids1, input_ids2, truth_ids[:, 1:], mask1, mask2, mask_truth[:, 1:], torch.LongTensor(labels)


def tokenize_gen(data, hps):
    if hps.model_name == 'bart':
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    elif hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir, padding_side='left')
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer = None

    inputs = []
    labels = []
    premise = []
    for example in data:
        if hps.model_name == 'bart':
            seq1 = example['cause'] + example['effect']
            seq2 = example['conceptual_explanation']
            inputs.append(seq1)
            labels.append(seq2)
        elif hps.model_name == 'gpt2':
            if hps.prompt == 'T0':
                inputs.append([example['cause'][:-1] + ' is the cause. ' + example['effect'][
                                                                           :-1] + ' is the effect. What is the explanation?',
                               example['conceptual_explanation']])
                premise.append(example['cause'][:-1] + ' is the cause. ' + example['effect'][
                                                                           :-1] + ' is the effect. What is the explanation?')
            elif hps.prompt == 'T1':
                inputs.append([example['cause'][:-1] + ' causes ' + example['effect'][:-1] + '. Why?',
                               example['conceptual_explanation']])
                premise.append(example['cause'][:-1] + ' causes ' + example['effect'][:-1] + '. Why?')
            elif hps.prompt == 'T2':
                inputs.append(['Cause: ' + example['cause'] + ' ' + 'Effect: ' + example['effect'] + ' Explanation: ',
                               example['conceptual_explanation']])
                premise.append('Cause: ' + example['cause'] + ' ' + 'Effect: ' + example['effect'] + ' Explanation: ')
            else:
                inputs.append([example['cause'] + ' ' + example['effect'], example['conceptual_explanation']])
                premise.append(example['cause'] + ' ' + example['effect'])
            labels.append(example['conceptual_explanation'])
        else:
            return

    if hps.model_name == 'bart':
        outputs = tokenizer(inputs, padding=True)
        input_ids = torch.LongTensor(outputs['input_ids'])
        input_attention_mask = torch.LongTensor(outputs['attention_mask'])
        label_output = tokenizer(labels, padding=True)
        label_ids = torch.LongTensor(label_output['input_ids'])
        label_attention_mask = torch.LongTensor(label_output['attention_mask'])

        return input_ids, input_attention_mask, label_ids, label_attention_mask

    elif hps.model_name == 'gpt2':
        evaluate_outputs = tokenizer(labels, padding=True, return_token_type_ids=True)
        labels_ids = torch.LongTensor(evaluate_outputs['input_ids'])
        labels_mask = torch.LongTensor(evaluate_outputs['attention_mask'])
        labels_seg_id = torch.LongTensor(evaluate_outputs['token_type_ids'])

        tokenizer.padding_side = 'left'
        outputs = tokenizer(inputs, padding=True, return_token_type_ids=True)
        input_ids = torch.LongTensor(outputs['input_ids'])
        input_attention_mask = torch.LongTensor(outputs['attention_mask'])
        input_seg_id = torch.LongTensor(outputs['token_type_ids'])

        premise_outputs = tokenizer(premise, padding=True, return_token_type_ids=True)
        premise_ids = torch.LongTensor(premise_outputs['input_ids'])
        premise_mask = torch.LongTensor(premise_outputs['attention_mask'])
        premise_seg_ids = torch.LongTensor(premise_outputs['token_type_ids'])
        return input_ids, input_attention_mask, input_seg_id, labels_ids, labels_mask, labels_seg_id, premise_ids, premise_mask, premise_seg_ids


"""
    Metric Computation
"""


def compute_ppl(hps, model, data):
    # device = 'cuda'
    if hps.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir, padding_side='left')
        lls = []
        total_length = 0
        for example in data:
            # input_text = example['cause'] + ' ' + example['effect']
            if hps.prompt == 'T0':
                input_text = example['cause'][:-1] + ' is the cause. ' + example['effect'][
                                                                         :-1] + ' is the effect. What is the explanation?'
            elif hps.prompt == 'T1':
                input_text = example['cause'][:-1] + ' causes ' + example['effect'][:-1] + '. Why?'
            elif hps.prompt == 'T2':
                input_text = 'Cause: ' + example['cause'] + ' ' + 'Effect: ' + example['effect'] + ' Explanation: '
            else:
                input_text = example['cause'] + ' ' + example['effect']
            truth = example['conceptual_explanation']
            inputs = tokenizer(input_text)
            input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0).cuda()
            attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0).cuda()
            label_inputs = tokenizer(truth)
            label_ids = torch.LongTensor(label_inputs['input_ids']).unsqueeze(0).cuda()
            length = label_ids.shape[1]
            total_length += length

            # label_mask = torch.LongTensor(label_inputs['attention_mask']).unsqueeze(0).cuda()
            attention_mask = torch.cat((attention_mask, torch.ones(1, label_ids.shape[1]).long().cuda()), 1)
            label_ids = torch.cat((torch.LongTensor([-100] * input_ids.shape[1]).unsqueeze(0).cuda(), label_ids), 1)
            input_ids = torch.cat((input_ids, label_ids[:, input_ids.shape[1]:]), 1)
            with torch.no_grad():
                loss = model(input_ids, attention_mask=attention_mask, labels=label_ids)[0]
                lls.append(loss * length)

        ppl = torch.exp(torch.stack(lls).sum() / total_length)

    else:
        tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')
        lls = []
        total_length = 0
        for example in data:
            input_text = example['cause'] + ' ' + example['effect']
            truth = example['conceptual_explanation']
            inputs = tokenizer(input_text)
            input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0).cuda()
            attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0).cuda()
            label_inputs = tokenizer(truth)
            label_ids = torch.LongTensor(label_inputs['input_ids']).unsqueeze(0).cuda()
            length = label_ids.shape[1]
            total_length += length
            label_mask = torch.LongTensor(label_inputs['attention_mask']).unsqueeze(0).cuda()
            # attention_mask = torch.cat((attention_mask, torch.ones(1, label_ids.shape[1]).long().cuda()), 1)
            # label_ids = torch.cat((torch.LongTensor([-100]*input_ids.shape[1]).unsqueeze(0).cuda(), label_ids), 1)
            # input_ids = torch.cat((input_ids, label_ids[:, input_ids.shape[1]:]), 1)
            with torch.no_grad():
                loss = model(input_ids, attention_mask=attention_mask, decoder_input_ids=label_ids,
                             decoder_attention_mask=label_mask, labels=label_ids)[0]
                lls.append(loss * length)

        ppl = torch.exp(torch.stack(lls).sum() / total_length)

    return ppl.item()


"""
    Evaluation
"""


def cl_evaluation(hps, dataloader, model, loss_function, eval_step, exp_path, mode='dev', print_pred=True,
                  verbose=False):
    predictions = []
    labels = []
    loss = 0
    model.eval()
    for batch in dataloader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        if mode == 'dev':
            sent, seg_id, atten_mask, tmp_label, tmp_length = batch
            batch_size = len(sent)
            probs_hypothesis_0, probs_hypothesis_1 = model(sent, atten_mask, tmp_label, seg_ids=seg_id,
                                                           length=tmp_length, mode='eval')
            reasoning_labels = tmp_label[:, 1:].reshape(-1)
            label = torch.argmax(reasoning_labels.view(-1, 2), dim=1).tolist()

        probs = torch.cat([probs_hypothesis_0, probs_hypothesis_1], dim=-1)
        loss += loss_function(probs.view(-1), reasoning_labels.float()).item()
        pred = torch.argmax(probs, dim=1).cpu()
        predictions += pred.tolist()
        labels += label
        if verbose:
            print("label:", label)
            print("pred:", pred)

    count = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            count += 1
        else:
            continue
    acc = count / len(predictions)

    if print_pred:
        print_prediction(exp_path, "contrastive_cr", hps, eval_step, predictions)

    return acc, loss


def siamese_cr_evaluation(hps, dataloader, model, loss_function, eval_step, exp_path, mode='dev', print_pred=True):
    predictions = []
    labels = []
    loss = 0
    model.eval()
    for batch in dataloader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        if mode == 'dev':
            sent, attention_mask, seg_id, batch_labels, length, ask_for = batch
            probs_hypothesis_0, probs_hypothesis_1 = model(sent, attention_mask, ask_for=ask_for, seg_ids=seg_id,
                                                           length=length, mode="eval")

        probs = torch.cat([probs_hypothesis_0, probs_hypothesis_1], dim=-1)
        pred = torch.argmax(probs, dim=1).cpu()
        predictions += pred.tolist()

        loss += loss_function(probs.view(-1), batch_labels.view(-1).float()).item()
        labels += torch.argmax(batch_labels, dim=1).cpu().tolist()

    assert len(predictions) == len(labels), "Predictions length should be equal to the labels length!"

    count = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            count += 1
        else:
            continue
    acc = count / len(predictions)

    if print_pred:
        print_prediction(exp_path, "siamese_cr", hps, eval_step, predictions)

    return acc, loss


def vanilla_cr_evaluation(hps, dataloader, model, loss_function, eval_step, exp_path, mode='dev', print_pred=True):
    predictions = []
    labels = []
    loss = 0
    model.eval()
    for batch in dataloader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        if mode == 'dev':
            sent, seg_id, atten_mask, tmp_labels, tmp_length = batch
            probs = model(sent, atten_mask, seg_ids=seg_id, length=tmp_length).squeeze()
        else:
            sent, atten_mask, tmp_labels = batch
            _, probs = model(sent, atten_mask)

        predictions += probs.squeeze().cpu().tolist()
        loss += loss_function(probs, tmp_labels.float()).item()
        labels += tmp_labels.cpu().numpy().tolist()

    if hps.data_name == 'commonsenseqa':
        a1 = torch.FloatTensor(predictions[::5]).unsqueeze(1)
        a2 = torch.FloatTensor(predictions[1::5]).unsqueeze(1)
        a3 = torch.FloatTensor(predictions[2::5]).unsqueeze(1)
        a4 = torch.FloatTensor(predictions[3::5]).unsqueeze(1)
        a5 = torch.FloatTensor(predictions[4::5]).unsqueeze(1)
        a = torch.cat((a1, a2, a3, a4, a5), dim=1)

        t_a1 = torch.FloatTensor(labels[::5]).unsqueeze(1)
        t_a2 = torch.FloatTensor(labels[1::5]).unsqueeze(1)
        t_a3 = torch.FloatTensor(labels[2::5]).unsqueeze(1)
        t_a4 = torch.FloatTensor(labels[3::5]).unsqueeze(1)
        t_a5 = torch.FloatTensor(labels[4::5]).unsqueeze(1)
        t_a = torch.cat((t_a1, t_a2, t_a3, t_a4, t_a5), dim=1)
        predict_labels = torch.argmax(a, 1).tolist()
        true_labels = torch.argmax(t_a, 1).tolist()

    elif hps.data_name == 'because':
        # softmax = nn.Softmax(1)
        a = predictions
        t_a = labels
        predict_labels = torch.sigmoid(torch.FloatTensor(a)).tolist()
        true_labels = t_a
        for k, p in enumerate(predict_labels):
            if p >= 0.5:
                predict_labels[k] = 1
            else:
                predict_labels[k] = 0

    elif hps.data_name == 'event_storyline':
        a = predictions
        predict_labels = torch.sigmoid(torch.FloatTensor(a)).tolist()
        predict_labels = [1 if p >= 0.5 else 0 for p in predict_labels]
        t_a = labels
        tp, tn, fp, fn = 0, 0, 0, 0
        for k in range(len(t_a)):
            if labels[k] == 1 and predict_labels[k] == 1:
                tp += 1
            elif labels[k] == 1 and predict_labels[k] == 0:
                fn += 1
            elif labels[k] == 0 and predict_labels[k] == 1:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, 0

    else:
        a1 = torch.FloatTensor(predictions[::2]).unsqueeze(1)
        a2 = torch.FloatTensor(predictions[1::2]).unsqueeze(1)
        a = torch.cat((a1, a2), dim=1)
        t_a1 = torch.FloatTensor(labels[::2]).unsqueeze(1)
        t_a2 = torch.FloatTensor(labels[1::2]).unsqueeze(1)
        t_a = torch.cat((t_a1, t_a2), dim=1)
        predict_labels = torch.argmax(a, 1).tolist()
        true_labels = torch.argmax(t_a, 1).tolist()

    count = 0
    for i in range(len(predict_labels)):
        if predict_labels[i] == true_labels[i]:
            count += 1
        else:
            continue
    if print_pred:
        print_prediction(exp_path, "cr", hps, eval_step, predict_labels)

    return count / len(true_labels), loss


# called in gpt2_generate.py
def gpt2_eg_evaluate(hps, data_loader, model, eval_step, exp_path, mode='dev', print_pred=True):
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir, padding_side='left')
    val_loss = 0
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    rouge1r, rouge2r, rougelr = 0, 0, 0
    rouge = Rouge()
    output_text = []
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for batch in data_loader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        input_ids, input_mask, input_seg_ids, gen_ids, gen_mask, _, premise_ids, premise_mask, premise_token_type_ids = batch  # dev
        tmp = torch.ones(gen_mask.shape).long()
        count_mask_length = torch.sum(tmp == gen_mask.cpu(), 1).squeeze().tolist()
        true_labels = None
        for j in range(input_ids.shape[0]):
            if true_labels is None:
                # true_labels = torch.cat((torch.ones(count_mask_length[j]).long(), input_ids[j, count_mask_length[j]:].cpu())).unsqueeze(0)
                true_labels = torch.cat(
                    (input_ids[j, :-count_mask_length[j]] * 0 - 100, input_ids[j, -count_mask_length[j]:])).unsqueeze(0)
            else:
                # true_labels = torch.cat((true_labels, torch.cat((torch.ones(count_mask_length[j]).long(), input_ids[j, count_mask_length[j]:].cpu())).unsqueeze(0)), 0)
                true_labels = torch.cat((true_labels, torch.cat(
                    (input_ids[j, :-count_mask_length[j]] * 0 - 100, input_ids[j, -count_mask_length[j]:])).unsqueeze(
                    0)), 0)

        output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg_ids, labels=true_labels)
        loss = output[0]
        val_loss += loss.item()

        # output = sample_sequence(model, length, device='cuda', context=premise_ids, batch_size=hps.batch_size, attention_mask=premise_mask, input_type='ids')
        generated = model.generate(input_ids=premise_ids,
                                   attention_mask=premise_mask,
                                   max_length=hps.length + premise_ids.shape[1],
                                   num_beams=5,
                                   early_stopping=True,
                                   do_sample=True,
                                   no_repeat_ngram_size=3,
                                   repetition_penalty=1.5
                                   )

        # generated = output[:, premise_ids.shape[1]:]
        # pdb.set_trace()
        generated = generated[:, premise_ids.shape[1]:]

        generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                          generated.cpu().tolist()]
        gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                     gen_ids.cpu().tolist()]
        input_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                      premise_ids]
        output_text += [[generated_text[i].split('.')[0] + '.'] for i in range(len(input_text))]

        for i in range(generated.shape[0]):
            # predict_tokens = tokenizer.convert_ids_to_tokens(generated[i])
            # generated_text = remove_special_tokens(tokenizer.convert_tokens_to_string(predict_tokens))

            # gold_tokens = tokenizer.convert_ids_to_tokens(gen_ids[i])
            # gold_text = remove_special_tokens(tokenizer.convert_tokens_to_string(gold_tokens))

            bleu1 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [1, 0, 0, 0])
            bleu2 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [0, 1, 0, 0])
            bleu3 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [0, 0, 1, 0])
            bleu4 += bleu([gold_text[i]], generated_text[i].split('.')[0] + '.', [0, 0, 0, 1])

            try:
                scores = rouge.get_scores(generated_text[i], gold_text[i])
                rouge1r += scores[0]['rouge-1']['r']
                rouge2r += scores[0]['rouge-2']['r']
                rougelr += scores[0]['rouge-l']['r']
            except:
                continue

    num_instances = (len(data_loader) - 1) * hps.batch_size + gen_ids.shape[0]

    ceq = CEQ_gen(hps, eval_step)

    evaluation_output = dict(
        val_loss=val_loss * 100,
        bleu1=bleu1,
        bleu2=bleu2,
        bleu3=bleu3,
        bleu4=bleu4,
        avg_bleu=sum([bleu1, bleu2, bleu3, bleu4]) / 4,
        rouge1=rouge1r,
        rouge2=rouge2r,
        rougel=rougelr,
        CEQ=ceq
    )
    for metric in evaluation_output.keys():
        if metric != "CEQ":
            evaluation_output[metric] /= num_instances

    if print_pred:
        print_prediction(exp_path, f"{hps.prompt}_eg", hps, eval_step, output_text)

    return evaluation_output


"""
    Misc.
"""


def evaluate_multi_task(model, dataloader_input, dataloader_output, hps):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    count = 0
    for batch1, batch2, t in zip(dataloader_input, dataloader_output, trange(len(dataloader_input))):
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch1 = tuple(term.to(device) for term in batch1)
            batch2 = tuple(term.to(device) for term in batch2)

        input_ids, attention_mask, labels = batch1
        decoder_ids, decoder_mask = batch2
        scores, _ = model(input_ids,
                          attention_mask,
                          decoder_ids,
                          decoder_mask,
                          labels,
                          mode='train')
        scores = torch.cat((scores[::2].unsqueeze(1), scores[1::2].unsqueeze(1)), 1)
        index = torch.argmax(scores, 1)
        predict_labels = index.cpu().tolist()
        labels = torch.cat((labels[::2].unsqueeze(1), labels[1::2].unsqueeze(1)), 1)
        labels = torch.argmax(labels, 1).cpu().tolist()
        for k in range(len(predict_labels)):
            if labels[k] == predict_labels[k]:
                count += 1
            else:
                continue

        input_ids = torch.cat((input_ids[::2].unsqueeze(1), input_ids[1::2].unsqueeze(1)), 1)
        input_ids = input_ids[range(input_ids.shape[0]), index, :]
        attention_mask = torch.cat((attention_mask[::2].unsqueeze(1), attention_mask[1::2].unsqueeze(1)), 1)
        attention_mask = attention_mask[range(attention_mask.shape[0]), index, :]

        # for i in range(input_ids.shape[0]):
        gen_ids = model(input_ids,
                        attention_mask,
                        decoder_ids,
                        decoder_mask,
                        labels,
                        mode='generate')
        generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                          gen_ids.tolist()]
        gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                     decoder_ids[::2, :].tolist()]

        for i in range(len(generated_text)):
            bleu1 += bleu([gold_text[i]], generated_text[i], [1, 0, 0, 0])
            bleu2 += bleu([gold_text[i]], generated_text[i], [0, 1, 0, 0])
            bleu3 += bleu([gold_text[i]], generated_text[i], [0, 0, 1, 0])
            bleu4 += bleu([gold_text[i]], generated_text[i], [0, 0, 0, 1])

    num_instances = (len(dataloader_output) - 1) * hps.batch_size // 2 + input_ids.shape[0]
    return count / num_instances, bleu1 / num_instances, bleu2 / num_instances, bleu3 / num_instances, bleu4 / num_instances


def evaluation_bart(dataloader, model, hps):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')
    score = 0
    for batch in dataloader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        input_ids, input_mask, labels, label_mask = batch
        predict_id = torch.zeros([input_ids.shape[0], 1]).long().cuda()
        decoder_ids = torch.zeros([input_ids.shape[0], 1]).long().cuda()

        while decoder_ids.shape[1] < 35 and predict_id.tolist() not in [[[2], [2]], [[1], [1]]]:
            output = model(input_ids, input_mask=input_mask, decoder_ids=decoder_ids, mode='test')
            predict_id = torch.argmax(output[0][:, -1, :], -1).unsqueeze(1)

            decoder_ids = torch.cat((decoder_ids, predict_id), -1)

        label_tokens = [tokenizer.convert_ids_to_tokens(labels[i]) for i in range(labels.shape[0])]
        predict_tokens = [tokenizer.convert_ids_to_tokens(decoder_ids[i]) for i in range(decoder_ids.shape[0])]
        references = [tokenizer.convert_tokens_to_string(tokens) for tokens in label_tokens]
        hypothesis = [tokenizer.convert_tokens_to_string(tokens) for tokens in predict_tokens]
        references = [remove_special_tokens(text) for text in references]
        hypothesis = [remove_special_tokens(text) for text in hypothesis]

        score += sum([bleu([references[i]], hypothesis[i]) for i in range(len(references))])

    return score / len(dataloader) / hps.batch_size


def evaluate_gpt2(dataloader, model, hps):
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir, padding_side='left')
    score = 0
    for batch in dataloader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        gen_ids, gen_mask, _, premise_ids, premise_mask, premise_token_type_ids = batch
        decode_ids = torch.zeros([premise_ids.shape[0], 1]).long().cuda()
        predict_id = torch.zeros([premise_ids.shape[0], 1]).long().cuda()

        while decode_ids.shape[1] <= 35 and predict_id.tolist() != (
                torch.ones([hps.batch_size, 1]).long() * 50256).tolist():
            output = model(premise_ids, premise_mask, token_type_ids=premise_token_type_ids, mode='test')
            predict_id = torch.argmax(output[1][:, -1, :], -1).unsqueeze(1)
            decode_ids = torch.cat((decode_ids, predict_id), -1)
            premise_ids = torch.cat((premise_ids, predict_id), -1)
            premise_mask = torch.cat((premise_mask, torch.ones([premise_mask.shape[0], 1]).long().cuda()), -1)
            premise_token_type_ids = torch.cat(
                (premise_token_type_ids, torch.ones([premise_token_type_ids.shape[0], 1]).long().cuda()), -1)

        label_tokens = [tokenizer.convert_ids_to_tokens(gen_ids[i]) for i in range(gen_ids.shape[0])]
        predict_tokens = [tokenizer.convert_ids_to_tokens(decode_ids[i][1:]) for i in range(decode_ids.shape[0])]
        references = [tokenizer.convert_tokens_to_string(tokens) for tokens in label_tokens]
        hypothesis = [tokenizer.convert_tokens_to_string(tokens) for tokens in predict_tokens]
        references = [remove_special_tokens(text) for text in references]
        hypothesis = [remove_special_tokens(text) for text in hypothesis]

        score += sum([bleu([references[i]], hypothesis[i]) for i in range(len(references))])

    return score / len(dataloader) / hps.batch_size


def remove_special_tokens(text):
    return text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').replace('<unk>', '').replace(
        '<|endoftext|>', '')


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=0.7, top_k=40,
                    device='cuda', sample=True, attention_mask=None, input_type='ids'):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        if input_type == 'ids':
            context = torch.tensor(context, device=device, dtype=torch.long)
        else:
            context = torch.tensor(context, device=device)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output_id = None
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            if input_type == 'ids':
                gen_output = model(input_ids=output, attention_mask=attention_mask, past_key_values=None, mode='test')
                logits = gen_output['logits']
            else:
                logits, past, hiddens = model(inputs_embeds=output, attention_mask=attention_mask, past_key_values=None,
                                              output_hidden_states=True)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            if input_type == 'ids':
                output = torch.cat((output, prev), dim=1)
            else:
                output = torch.cat((output, hiddens[-1][:, -1, :].unsqueeze(1)), 1)
                output_id = prev if output_id is None else torch.cat((output_id, prev), 1)

            attention_mask = torch.cat((attention_mask, torch.ones(prev.shape).long().cuda()), -1)
    return output if input_type == 'ids' else output_id


def bart_evaluate(model, data_loader, hps):
    tokenizer = BartTokenizer.from_pretrained(hps.model_dir, padding_side='left')

    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    rouge1p, rouge1r, rouge1f, rouge2p, rouge2r, rouge2f, rougelp, rougelr, rougelf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    # rouge1p, rouge1r, rouge1f, rouge2p, rouge2r, rouge2f, rougelp, rougelr, rougelf = 0, 0, 0, 0, 0, 0, 0, 0, 0
    rouge = Rouge()
    nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_text = []

    for batch in data_loader:
        if hps.cuda:
            device = f"cuda:{hps.gpu}"
            batch = tuple(term.to(device) for term in batch)

        input_ids, input_mask, labels, label_mask = batch
        generate_ids = model.generate(input_ids,
                                      attention_mask=input_mask,
                                      num_beams=hps.beam_size,
                                      max_length=hps.length,
                                      early_stopping=True,
                                      no_repeat_ngram_size=3,
                                      repetition_penalty=1.5,
                                      # temperature=0.7,
                                      # length_penalty=0.6
                                      )
        # generate_ids = generate_ids[:, input_ids.shape[1]:]

        generate_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                         generate_ids]
        gold_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in labels]
        input_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                      input_ids]

        output_text += [[input_text[i], gold_text[i], generate_text[i].split('.')[0] + '.'] for i in
                        range(len(input_text))]

        for i in range(len(gold_text)):

            bleu1 += bleu([gold_text[i]], generate_text[i].split('.')[0] + '.', [1, 0, 0, 0])
            bleu2 += bleu([gold_text[i]], generate_text[i].split('.')[0] + '.', [0, 1, 0, 0])
            bleu3 += bleu([gold_text[i]], generate_text[i].split('.')[0] + '.', [0, 0, 1, 0])
            bleu4 += bleu([gold_text[i]], generate_text[i].split('.')[0] + '.', [0, 0, 0, 1])

            try:
                scores = rouge.get_scores(generate_text[i], gold_text[i])
            except:
                scores = [
                    {
                        "rouge-1": {
                            "f": 0.0,
                            "p": 0.0,
                            "r": 0.0
                        },
                        "rouge-2": {
                            "f": 0.0,
                            "p": 0.0,
                            "r": 0.0
                        },
                        "rouge-l": {
                            "f": 0.0,
                            "p": 0.0,
                            "r": 0.0
                        }
                    }
                ]
            rouge1 = scores[0]['rouge-1']
            rouge1f += rouge1['f']
            rouge1p += rouge1['p']
            rouge1r += rouge1['r']

            rouge2 = scores[0]['rouge-2']
            rouge2f += rouge2['f']
            rouge2p += rouge2['p']
            rouge2r += rouge2['r']

            rougel = scores[0]['rouge-l']
            rougelf += rougel['f']
            rougelp += rougel['p']
            rougelr += rougel['r']

    num_instances = (len(data_loader) - 1) * hps.batch_size + input_ids.shape[0]

    fo = open(hps.output_dir + '/bart_predict_' + nowtime + '.csv', 'w', encoding='utf-8')
    writer = csv.writer(fo)
    writer.writerows(output_text)

    return bleu1 / num_instances, bleu2 / num_instances, bleu3 / num_instances, bleu4 / num_instances, rouge1r / num_instances, rouge2r / num_instances, rougelr / num_instances


"""
    CEQ Metric.
"""


def tokenize_ceq(sent):
    sent = sent.lower()
    sent = sent.strip('.')
    lemmatizer = ns.WordNetLemmatizer()
    sent = sent.replace("'s", '')
    sent = sent.split(' ')
    for ith, word in enumerate(sent):
        word_n = lemmatizer.lemmatize(word, pos='n')
        word_v = lemmatizer.lemmatize(word, pos='v')

        if word_n != word_v:
            if word_n == word:
                sent[ith] = word_v
            else:
                sent[ith] = word_n
    return sent


def cs_word_ceq(w_cause, w_effect, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA):
    M = 62675002

    try:
        p_w_cause = float(sum(causes[w_cause].values())) / M
    except KeyError:
        p_w_cause = 0

    try:
        p_w_effect = float(sum(effects[w_effect].values())) / M
    except KeyError:
        p_w_effect = 0

    try:
        p_join = float(causes[w_cause][w_effect]) / M
    except KeyError:
        p_join = 0

    if p_join != 0:
        cs_nes = p_join / p_w_cause ** ALPHA / p_w_effect
        cs_surf = p_join / p_w_cause / p_w_effect ** ALPHA
        cs = cs_nes ** LAMBDA * cs_surf ** (1 - LAMBDA)
    else:
        cs = float(2) / len(causes)
    return cs


def cs_sent_ceq(s_cause, s_effect, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA):
    cs = 0
    num_zero = 0
    for w_cause in s_cause:
        for w_effect in s_effect:
            cs_tmp = cs_word_ceq(w_cause, w_effect, effects, causes, ALPHA=ALPHA, LAMBDA=LAMBDA)
            cs = cs + cs_tmp
            if cs_tmp == 0:
                num_zero = num_zero + 1
    cs = cs / (len(s_cause) + len(s_effect))

    return cs


def inf_ceq(data, i, hps, causes, effects):
    L = data.shape[0]

    premise = data['cause'].tolist()
    hypothesis = data['effect'].tolist()
    truth = data['explanation'].tolist()

    pred = list()
    reference_rnn = list()
    reference_gpt2 = list()
    reference_mt = list()

    # Causal_Strength(Cause, Effect)
    cs_1 = list()

    # Causal_Strength(Cause + Truth, Effect)
    cs_2 = list()

    # Causal_Strength(Cause, Truth + Effect)
    cs_3 = list()

    # CEQ result
    cs = list()

    cnt = 0
    r = 0

    for ith in trange(L):
        premise_tmp = tokenize_ceq(premise[ith])
        hypothesis_tmp = tokenize_ceq(hypothesis[ith])
        premise_truth_tmp = tokenize_ceq(premise[ith] + truth[ith])
        truth_hypothesis_tmp = tokenize_ceq(truth[ith] + hypothesis[ith])

        cs_tmp_1 = cs_sent_ceq(premise_tmp, hypothesis_tmp, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)
        cs_tmp_2 = cs_sent_ceq(premise_truth_tmp, hypothesis_tmp, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)
        cs_tmp_3 = cs_sent_ceq(premise_tmp, truth_hypothesis_tmp, causes, effects, ALPHA=ALPHA, LAMBDA=LAMBDA)

        cs_1.append(cs_tmp_1)
        cs_2.append(cs_tmp_2)
        cs_3.append(cs_tmp_3)
        cs.append(max(cs_tmp_2, cs_tmp_3) - cs_tmp_1)
        cnt += 1
        r += max(cs_tmp_2, cs_tmp_3) - cs_tmp_1

    res = data.copy()
    res['Cause:Effect'] = cs_1
    res['Cause+Truth:Effect'] = cs_2
    res['Cause:Truth+Effect'] = cs_3
    res['CEQ'] = cs

    res.to_csv(f"/data/data_CEQ/{hps.prompt}_CEQ_epoch{i}.csv")
    print("Average CEQ for " + str(i) + " is: ", (r / cnt))
    return (r / cnt)


def CEQ_gen(hps, epoch):
    dev_data = [json.loads(line) for line in open('../../data/Explanation_Generation/dev.jsonl', 'r')]
    headerList = ['cause', 'effect', 'explanation']

    with open('/data/data_CEQ/ceq_data1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile)
        for d in dev_data:
            c = d['cause']
            e = d['effect']
            spamwriter.writerow([c, e])

    with open('/data/data_CEQ/ceq_data1.csv') as in_1, open(f"/data/output/saved_model/{hps.prompt}_generate_gpt2/predictions/{hps.prompt}_eg_pred_epoc_{epoch}.csv") as in_2, open(
            '/data/data_CEQ/ceq_data.csv', 'w') as out:
        reader1 = csv.reader(in_1)
        reader2 = csv.reader(in_2)
        writer = csv.writer(out)
        writer.writerow(headerList)
        for row1, row2 in zip(reader1, reader2):
            if row1[0] and row2[0]:
                writer.writerow([row1[0], row1[1], row2[0]])

    data = pd.read_csv('/data/data_CEQ/ceq_data.csv')

    f = open("/data/data_CEQ/causes.pkl", 'rb')
    causes = pickle.load(f)
    f.close()

    f = open("/data/data_CEQ/effects.pkl", 'rb')
    effects = pickle.load(f)
    f.close()

    res = inf_ceq(data, epoch, hps, causes, effects)