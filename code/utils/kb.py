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
    return torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0), \
        torch.stack(segment_ids, dim=0), torch.stack(soft_pos_ids, dim=0), torch.LongTensor(labels)


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
                soft_pos_id = soft_pos_id + 2

            features.append(
                ((tokens, input_ids, attention_mask, segment_ids, soft_pos_id), labels[i]))
    return features


def add_knowledge_with_vm(mp_all,
                          sent_batch,
                          tokenizer,
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
                                '/r/CausesDesire': 'causes the desire of',
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
                                '/r/HasPrerequisite': 'has prerequisite is',
                                '/r/HasProperty': 'has an attribute is',
                                '/r/HasSubevent': 'has a subevent is',
                                '/r/InstanceOf': 'runs an instance of',
                                '/r/IsA': 'is a',
                                '/r/LocatedNear': 'is located near',
                                '/r/MadeOf': 'is made of',
                                '/r/MannerOf': 'is the manner of',
                                '/r/MotivatedByGoal': 'is a step toward accomplishing the goal',
                                '/r/NotCapableOf': 'is not capable of',
                                '/r/NotDesires': 'does not desire',
                                '/r/NotHasProperty': 'has no attribute',
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

    split_sent_batch = [tokenizer.tokenize(sent) for sent in sent_batch]
    know_sent_batch = []
    position_batch = []
    visible_matrix_batch = []
    seg_batch = []
    for split_sent in split_sent_batch:

        # create tree
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1  # soft position idx，深度相同的节点 idx 相等
        abs_idx = -1  # hard position idx，不重复
        abs_idx_src = []
        for token in split_sent:
            """
            k-bert 这里只挑了前 max_entities 个 kg 里邻接的实体，如果采样得出或根据其他方法会不会更好
            """
            # entities = list(mp_all.get(token,
            #                            []))[:max_entities]
            # Ġ 是 GPT-2/Roberta Tokenizer，▁ 是 Albert 中的
            entities = sorted(list(mp_all.get(token.strip(',|.|?|;|:|!|Ġ|_|▁'), [])), key=lambda x: x[2], reverse=True)[
                       :max_entities]

            sent_tree.append((token, entities))

            if token in tokenizer.all_special_tokens:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            else:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
                # token_pos_idx = [
                #     pos_idx + i for i in range(1,
                #                                len(token) + 1)
                # ]
                # token_abs_idx = [
                #     abs_idx + i for i in range(1,
                #                                len(token) + 1)
                # ]
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                ent_values = conceptnet_relation_to_nl(ent)

                ent_pos_idx = [
                    token_pos_idx[-1] + i for i in range(1,
                                                         len(ent_values) + 1)
                ]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(ent_values) + 1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []  # 每个 token 占一个
        pos = []  # 每个 token 的 soft position idx
        seg = []  # token 是属于主干还是分支，主干为 0，分支为 1
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in tokenizer.all_special_tokens:
                know_sent += [word]
                seg += [0]
            else:
                know_sent += [word]
                seg += [0]
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                ent = sent_tree[i][1][j]  # ('university', '/r/AtLocation', 6.325)
                ent_values = conceptnet_relation_to_nl(ent)

                add_word = ent_values
                know_sent += add_word
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])

        token_num = len(know_sent)

        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                # abs_idx_src 代表所有主干上的节点 id，src_ids 为当前遍历主干 token 的 id
                # 这里 visible_abs_idx 代表主干上的节点可以看到主干其他节点，并且也可以看到其下面分支的节点
                visible_abs_idx = abs_idx_src + [
                    idx for ent in item[1] for idx in ent
                ]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    # 这里遍历分支节点，它可以看到该分支上所有节点以及其依赖的那些主干节点
                    # 依赖的主干节点可能有多个，因为一个词比如 “我的世界” 它分字后有四个节点
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [tokenizer.pad_token] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(visible_matrix,
                                    ((0, pad_num), (0, pad_num)),
                                    'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)

    return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


def create_datasets_with_kbert(features, shuffle=True):
    """
    使用 features 构建 dataset
    :param features:
    :param choices_num: 选项(label) 个数
    :param shuffle: 是否随机顺序，默认 True
    :return:
    """
    if shuffle:
        perm = torch.randperm(len(features))
        features = [features[i] for i in perm]
    x = [i[0] for i in features]
    y = torch.tensor([i[1] for i in features])
    return MyDataset(x, y)