import pandas as pd
import numpy as np
import pickle
import nltk.stem as ns
import pdb
from tqdm import trange
import os
import csv
import json

ALPHA = 0.66
LAMBDA = 1
EG_filename = '/data/output/saved_model/T0_generate_gpt2/predictions/T0_eg_pred_step_'
OUT_filename = '/data/data_CEQ/T0_CEQ_'

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
        
    # print(p_w_cause, p_w_effect, p_join)
    
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
    
    
def inf_ceq(data, i):
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
    
    res.to_csv(OUT_filename + str(i) + '.csv')
    print("Average CEQ for " + str(i) + " is: ", (r / cnt))
    return res


if __name__ == '__main__':
    dev_data = [json.loads(line) for line in open('../data/Explanation_Generation/dev.jsonl', 'r')]
    headerList = ['cause', 'effect', 'explanation']

    with open('/data/data_CEQ/ceq_data1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile)
        for d in dev_data:
            c = d['cause']
            e = d['effect']
            spamwriter.writerow([c, e])

    for i in range(10):
        with open('/data/data_CEQ/ceq_data1.csv') as in_1, open(EG_filename + str(i) + '.csv') as in_2, open('/data/data_CEQ/ceq_data.csv', 'w') as out:
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

        res = inf_ceq(data, i)
