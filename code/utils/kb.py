import pandas as pd
import torch
import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
from utils.graph import GraphUtils
from utils.utils import load_pretrained_tokenizer
from utils.kb_dataset import MyDataset

def get_all_features(data, hps, max_seq_length=128):
    semantic_features = get_features_with_kbert(data, hps,
                                              max_seq_length=max_seq_length)
    input_ids = [i[0][1] for i in semantic_features]
    attention_mask = [i[0][2] for i in semantic_features]
    segment_ids = [i[0][3] for i in semantic_features]
    soft_pos_ids = [i[0][4] for i in semantic_features]
    labels = [i[1] for i in semantic_features]  # 分离标签
    return torch.vstack(input_ids), torch.vstack(attention_mask), \
        torch.vstack(segment_ids), torch.vstack(soft_pos_ids), torch.LongTensor(labels)


def get_features_with_kbert(data, hps,
                                        max_seq_length):
    tokenizer = load_pretrained_tokenizer(hps)

    graph = GraphUtils()
    print('graph init...')
    graph.load_mp_all_by_pickle(graph.args['mp_pickle_path'])
    # graph.init(is_load_necessary_data=True)
    print('merge graph by downgrade...')
    graph.merge_graph_by_downgrade()
    print('reduce graph noise...')
    graph.reduce_graph_noise()  # 根据黑白名单，停用词，边权等信息进行简单修剪
    print('reduce graph noise done!')

    features = []
    for example in data:
        premise, a1, a2 = example['premise'], example['hypothesis1'], example['hypothesis2']
        if example['ask-for'] == 'cause':
            instance1 = [a1, premise]
            instance2 = [a2, premise]
        else:
            instance1 = [premise, a1]
            instance2 = [premise, a2]

        # choices_features = []
        labels = [0, 1] if example['label'] == 1 else [1, 0]
        for i, instance in enumerate([instance1, instance2]):
            context_tokens = instance[0]
            ending_tokens = instance[1]

            source_sent = '{} {} {} {} {}'.format(tokenizer.cls_token,
                                                  context_tokens,
                                                  tokenizer.sep_token,
                                                  ending_tokens,
                                                  tokenizer.sep_token)

            tokens, soft_pos_id, attention_mask, segment_ids = add_knowledge_with_vm(mp_all=graph.mp_all,
                                                                                     sent_batch=[source_sent],
                                                                                     tokenizer=tokenizer,
                                                                                     max_entities=2,
                                                                                     max_length=max_seq_length)
            tokens = tokens[0]
            soft_pos_id = torch.LongTensor(soft_pos_id[0])
            attention_mask = torch.LongTensor(attention_mask[0])
            segment_ids = torch.LongTensor(segment_ids[0])
            input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens))

            assert input_ids.shape[0] == max_seq_length
            assert attention_mask.shape[0] == max_seq_length
            assert soft_pos_id.shape[0] == max_seq_length
            assert segment_ids.shape[0] == max_seq_length

            # soft_pos_id = soft_pos_id[0]
            # attention_mask = attention_mask[0]
            # segment_ids = segment_ids[0]
            # input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # assert len(input_ids) == max_seq_length
            # assert len(attention_mask) == max_seq_length
            # assert len(soft_pos_id) == max_seq_length
            # assert len(segment_ids) == max_seq_length

            if 'Roberta' in str(type(tokenizer)):
                # 这里做特判是因为 Roberta 的 Embedding pos_id 是从 2 开始的
                # 而 Bert 是从零开始的
                soft_pos_id = s