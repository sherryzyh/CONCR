import pandas as pd
import torch
import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
from utils.graph import GraphUtils
from utils.utils import load_pretrained_tokenizer
from utils.kb_dataset import MyDataset
import spacy

def get_all_features(data, hps, nlp, max_seq_length=128):
    semantic_features = get_features_with_kbert(data, hps, nlp,
                                              max_seq_length=max_seq_length)
    input_ids = [i[0][1] for i in semantic_features]
    attention_mask = [i[0][2] for i in semantic_features]
    segment_ids = [i[0][3] for i in semantic_features]
    soft_pos_ids = [i[0][4] for i in semantic_features]
    labels = [i[1] for i in semantic_features]  # 分离标签
    return torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0), \
        torch.stack(segment_ids, dim=0), torch.stack(soft_pos_ids, dim=0), torch.LongTensor(labels)


def get_features_with_kbert(data, hps, nlp, max_seq_length):
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
    # for example in data:
    for e, example in enumerate(data):
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
            cause, effect = instance

            # source_sent = '{} {} {} {} {}'.format(tokenizer.cls_token,
            #                                       context_tokens,
            #                                       tokenizer.sep_token,
            #                                       ending_tokens,
            #                                       tokenizer.sep_token)

            tokens, soft_pos_id, attention_mask, segment_ids = add_knowledge_with_vm(mp_all=graph.mp_all,
                                                                                     cause=cause,
                                                                                     effect=effect,
                                                                                     tokenizer=tokenizer,
                                                                                     nlp=nlp,
                                                                                     max_entities=2,
                                                                                     max_length=max_seq_length)
            soft_pos_id = torch.LongTensor(soft_pos_id)
            attention_mask = torch.LongTensor(attention_mask)
            segment_ids = torch.LongTensor(segment_ids)
            input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens))

            assert input_ids.shape[0] == max_seq_length, f"{input_ids.shape[0]}"
            assert attention_mask.shape[0] == max_seq_length, f"{attention_mask.shape[0]}"
            assert soft_pos_id.shape[0] == max_seq_length, f"{soft_pos_id[0]}"
            assert segment_ids.shape[0] == max_seq_length, f"{segment_ids[0]}"

            if 'Roberta' in str(type(tokenizer)):
                # 这里做特判是因为 Roberta 的 Embedding pos_id 是从 2 开始的
                # 而 Bert 是从零开始的
                soft_pos_id = soft_pos_id + 2

            features.append(
                ((tokens, input_ids, attention_mask, segment_ids, soft_pos_id), labels[i]))
            # if e < 5:
            #     print(i)
            #     print(f'tokens: {[(ti, t) for ti, t in enumerate(tokens)]}')
            #     print(f'input_ids: {input_ids}')
            #     print(f'attention_mask: \n1: {attention_mask[1]}\n5: {attention_mask[5]}\n15: {attention_mask[15]}\n20: {attention_mask[20]}')
            #     print(f'segment_ids: {segment_ids}')
            #     print(f'soft_pos_id: {soft_pos_id}')
            #     print(f'label: {labels[i]}')
    return features


def add_knowledge_with_vm(mp_all,
                          cause,
                          effect,
                          tokenizer,
                          nlp,
                          max_entities=2,
                          max_length=128):
    """
    input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
    return: know_sent_batch - list of sentences with entites embedding
            position_batch - list of position index of each character.
            visible_matrix_batch - list of visible matrixs
            seg_batch - list of segment tags
    """

    def conceptnet_relation_to_nl(ent):
        """
        :param ent: ('university', '/r/AtLocation', 6.325)
        :return: 返回 ent 翻译成自然语言并分词后的结果
        """
        relation_to_language = {'/r/AtLocation': 'is at the location of the',
                                '/r/CapableOf': 'is capable of',
                                '/r/Causes': 'causes',
                                '/r/CausesDesire': 'causes the desire for', # 'causes the desire of'
                                '/r/CreatedBy': 'is created by',
                                '/r/DefinedAs': 'is defined as',
                                '/r/DerivedFrom': 'is derived from',
                                '/r/Desires': 'desires',
                                '/r/Entails': 'entails',
                                '/r/EtymologicallyDerivedFrom': 'is etymologically derived from',
                                '/r/EtymologicallyRelatedTo': 'is etymologically related to',
                                '/r/FormOf': 'is an inflected form of',
                                '/r/HasA': 'has a',
                                '/r/HasContext': 'appears in the context of',
                                '/r/HasFirstSubevent': 'is an event that begins with subevent',
                                '/r/HasLastSubevent': 'is an event that concludes with subevent',
                                '/r/HasPrerequisite': 'has prerequisite of', # 'has prerequisite is'
                                '/r/HasProperty': 'has an attribute of', # 'has an attribute is'
                                '/r/HasSubevent': 'has a subevent is',
                                '/r/InstanceOf': 'runs an instance of',
                                '/r/IsA': 'is a',
                                '/r/LocatedNear': 'is located near',
                                '/r/MadeOf': 'is made of',
                                '/r/MannerOf': 'is the manner of',
                                '/r/MotivatedByGoal': 'is a step toward accomplishing the goal',
                                '/r/NotCapableOf': 'is not capable of',
                                '/r/NotDesires': 'does not desire',
                                '/r/NotHasProperty': 'has no attribute of', # 'has no attribute'
                                '/r/PartOf': 'is a part of',
                                '/r/ReceivesAction': 'receives action for',
                                '/r/RelatedTo': 'is related to',
                                '/r/SimilarTo': 'is similar to',
                                '/r/SymbolOf': 'is the symbol of',
                                '/r/UsedFor': 'is used for',
                                }
        # 这里加入一个 i，主要是为了让后面的作为非开头出现
        # ent_values = 'i {}'.format(ent[0].replace('_', ' '))
        ent_values = 'i {} {}'.format(relation_to_language.get(ent[1], ''),
                                      ent[0].replace('_', ' '))
        ent_values = tokenizer.tokenize(ent_values)[1:]

        # is_bpe_tokenizer = tokenizer.cls_token == '<s>'  # 适用于 Roberta/GPT
        # if is_bpe_tokenizer:
        #     # 因为这里是分支节点，因此是有空格分割的，针对 BPE 算法的分词器应该添加 Ġ
        #     ent_values[0] = 'Ġ' + ent_values[0]
        return ent_values

    cause_tokens = tokenizer.tokenize(cause)
    effect_tokens = tokenizer.tokenize(effect)
    instance_tokens = [tokenizer.cls_token] + cause_tokens + [tokenizer.sep_token] + effect_tokens + [tokenizer.sep_token]
    ori_token_type_ids = [0] * (len(cause_tokens) + 2) + [1] * (len(effect_tokens) + 1)
    # know_search: 
    #   key: index of ending token
    #   value: [index of starting token, word, tokenized_ent_list]
    know_search = dict()
    incomplete_word = False
    word_cache = ""
    cache_start = 0
    for i, token in enumerate(instance_tokens):
        if len(token) < 3 or token[:2] != "##":
            if not incomplete_word:
                word_cache = token
                continue
            know_search[i - 1] = [cache_start, word_cache]
            cache_start = i
            incomplete_word = False
            word_cache = token
        else:
            word_cache += token[2:]
            incomplete_word = True
    know_search[len(instance_tokens) - 1] = [cache_start, word_cache]
    ent_token_num = 0

    for key, value in know_search.items():
        query = nlp(value[1])[0].lemma_
        entities = sorted(list(mp_all.get(query.strip(',|.|?|;|:|!|Ġ|_|▁'), [])), key=lambda x: x[2], reverse=True)[
                    :max_entities]
        tokenized_ent_list = []
        for ent in entities:
            ent_tokens = conceptnet_relation_to_nl(ent)
            tokenized_ent_list.append(ent_tokens)
            ent_token_num += len(ent_tokens)
        know_search[key].append(tokenized_ent_list)
    
    know_sent = []  # 每个 token 占一个
    pos = []  # 每个 token 的 soft position idx
    seg = []  # token 是属于主干还是分支，主干为 0，分支为 1
    token_type_ids = [] # whether a token is in the first (0) or second (1) sentence
    new_sent_len = ent_token_num + len(instance_tokens)
    visible_matrix = np.zeros((new_sent_len, new_sent_len))
    hard_pos_idx = -1
    for i, ori_token in enumerate(instance_tokens):
        token_type = ori_token_type_ids[i]
        hard_pos_idx += 1
        know_sent.append(ori_token)
        pos.append(i)
        seg.append(0)
        token_type_ids.append(token_type)
        if i not in know_search:
            continue
        tokenized_ent_list = know_search[i][2]
        for ent in tokenized_ent_list:
            know_sent += ent
            pos += [(i + j + 1) for j in range(len(ent))]
            seg += [1] * len(ent)
            token_type_ids += [token_type] * len(ent)
            for j in range(hard_pos_idx + 1, hard_pos_idx + len(ent) + 1):
                for k in range(hard_pos_idx + 1, hard_pos_idx + len(ent) + 1):
                    visible_matrix[j][k] = 1
                visible_matrix[j][i] = 1
                visible_matrix[i][j] = 1
            hard_pos_idx += len(ent)
    for i, seg_val_i in enumerate(seg):
        for j, seg_val_j in enumerate(seg):
            if seg_val_i == 0 and seg_val_j == 0:
                visible_matrix[i][j] = 1
    src_length = len(know_sent)
    if len(know_sent) < max_length:
        pad_num = max_length - src_length
        know_sent += [tokenizer.pad_token] * pad_num
        seg += [0] * pad_num
        token_type_ids += [0] * pad_num
        pos += [max_length - 1] * pad_num
        visible_matrix = np.pad(visible_matrix,
                                ((0, pad_num), (0, pad_num)),
                                'constant')  # pad 0
    else:
        know_sent = know_sent[:max_length]
        seg = seg[:max_length]
        token_type_ids = token_type_ids[:max_length]
        pos = pos[:max_length]
        visible_matrix = visible_matrix[:max_length, :max_length]
    return know_sent, pos, visible_matrix, seg