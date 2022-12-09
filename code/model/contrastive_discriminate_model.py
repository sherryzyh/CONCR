from turtle import forward
import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
from transformers import OpenAIGPTConfig, OpenAIGPTModel, XLNetConfig, XLNetModel
from transformers import BartConfig, BartForSequenceClassification, BartModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from .discriminate_model import pretrained_model


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.dense.weight)
        self.dense.bias.data.fill_(0.01)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class CosSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        return self.cos(x, y) / self.temp


class Scorer(nn.Module):
    """
    Causal Scorer
    """

    def __init__(self, config, temp=0.05, dropout=0.3):
        super().__init__()
        self.temp = temp
        self.score_mlp = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        self.init_weights()

    def forward(self, x, y):
        # x, y [batch_size, hidden_size]
        pair = self.pairxy(x, y)
        score = self.score_mlp(pair).squeeze()
        return score / self.temp

    def init_weights(self):
        for m in self.score_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def pairxy(self, x, y):
        batch_size = x.size(0)
        hidden_size = x.size(1)
        idx_0 = torch.arange(batch_size)
        idx_1 = torch.arange(batch_size)
        idx_0, idx_1 = torch.meshgrid(idx_0, idx_1)
        # after meshgrid
        # idx_0 is [[0, 0], [1, 1]]
        # idx_1 is [[0, 1], [0, 1]]

        idx_0 = idx_0.reshape(-1)
        idx_1 = idx_1.reshape(-1)
        xypair = torch.cat([x[idx_0], y[idx_1]], dim=1)  # [batch_size * batch_size, hidden_size * 2]
        xypair = xypair.view(batch_size, batch_size, -1)  # [batch_size, batch_size, hidden_size * 2]
        # print("xypair.size:", xypair.size())
        return xypair


class contrastive_reasoning_model(nn.Module):
    def __init__(self, hps):
        super(contrastive_reasoning_model, self).__init__()
        self.hps = hps
        self.model_name = hps.model_name
        self.model_type = "contrastive"
        # self.discriminate_model = pretrained_model(hps)

        if hps.model_name == 'bert':
            self.sentence_encoder = BertModel.from_pretrained(hps.model_dir)
            self.config = BertConfig(hps.model_dir)
        elif hps.model_name == 'xlnet':
            self.sentence_encoder = XLNetModel.from_pretrained(hps.model_dir, mem_len=1024)
            self.config = XLNetConfig.from_pretrained(hps.model_dir)
        elif hps.model_name == 'bart':
            self.sentence_encoder = BartModel.from_pretrained(hps.model_dir)
            self.config = BartConfig.from_pretrained(hps.model_dir)

        # Scorer
        if hps.score == "cossim":
            self.sim = CosSimilarity()
        elif hps.score == "causalscore":
            self.sim = Scorer(self.config)

        # Contrastive Loss
        self.contrastive_loss = nn.CrossEntropyLoss()

    # compose causal pairs
    def compose_causal_pair(self, premise, hypothesis, labels, device):
        """
        Arguments:
            premise     [batch_size, hidden_size]
            hypothesis  [batch_size, hidden_size]
            labels     [batch_size, 3]
            device
        """
        batch_size = labels.size(0)
        causes = torch.zeros((batch_size, self.config.hidden_size))
        effects = torch.zeros((batch_size, self.config.hidden_size))

        # print("compose pair, batch_size:", batch_size)
        # print("compose pair, labels.size:", labels.size())
        for i in range(batch_size):
            if labels[i, 0] == 0:  # 'ask-for cause'
                # causal_pairs[i] = torch.concat([hypothesis[i], premise[i]], dim=-1)
                causes[i] = hypothesis[i]
                effects[i] = premise[i]
            else:  # 'ask-for effect'
                # causal_pairs[i] = torch.concat([premise[i], hypothesis[i]], dim=-1)
                causes[i] = premise[i]
                effects[i] = hypothesis[i]
        return causes.to(device), effects.to(device)

    def forward(self, input_ids, attention_mask, labels, seg_ids=None, length=None, mode='train'):
        if mode == 'train':
            return self.cl_forward(input_ids, attention_mask, labels, seg_ids, length)
        else:
            return self.eval_forward(input_ids, attention_mask, labels, seg_ids, length)

    def forward_sent_encoding(self, input_ids, attention_mask, labels, seg_ids=None, length=None):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if seg_ids is not None:
            seg_ids = seg_ids.view((-1, seg_ids.size(-1)))
        if length is not None:
            length = length.view(-1)

        if self.hps.model_name in ['bert', 'albert', 'gpt']:
            sent_embs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                              token_type_ids=seg_ids)
        else:
            sent_embs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Pooling
        # by default, use the "cls" embedding as the sentence representation
        if self.hps.model_name == "bert":
            pooler_output = sent_embs.pooler_output
        elif self.hps.model_name == "xlnet":
            pooler_output = sent_embs.last_hidden_state[:, 0, :]
        elif self.hps.model_name == "bart":
            pooler_output = sent_embs.encoder_last_hidden_state[:, 0, :]

        # print("pooler_output.size:", pooler_output.size())
        pooler_output = pooler_output.view(
            (batch_size, num_sent, pooler_output.size(-1)))  # [bs, num_sent, hidden_size]
        return pooler_output

    def cl_forward(self, input_ids, attention_mask, labels, seg_ids=None, length=None):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        device = input_ids.device

        # Sentence pooler encoding
        pooler_output = self.forward_sent_encoding(input_ids, attention_mask, labels, seg_ids, length)  # [bs, num_sent, hidden_size]

        # Separate representation
        premise, hypothesis_0 = pooler_output[:, 0], pooler_output[:, 1]
        causes_0, effects_0 = self.compose_causal_pair(premise, hypothesis_0, labels, device)

        contrastive_causal_score = self.sim(causes_0, effects_0)
        # print("contrastive score:", contrastive_causal_score.size())
        # print(contrastive_causal_score)

        # Hard negative
        if num_sent == 3:
            hypothesis_1 = pooler_output[:, 2]
            causes_1, effects_1 = self.compose_causal_pair(premise, hypothesis_1, labels, device)
            hardneg_causal_score = self.sim(causes_1, effects_1)

            # print("contrastive score:", contrastive_causal_score.size())
            # print("hardneg_causal_score:", hardneg_causal_score.size())

            contrastive_causal_score = torch.cat([contrastive_causal_score, hardneg_causal_score], dim=1)

            # Calculate loss with hard negatives
            # Note that weights are actually logits of weights

            # Add hard negative
            hard_neg_weight = self.hps.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (contrastive_causal_score.size(-1) - hardneg_causal_score.size(-1)) + [0.0] * i + [
                    hard_neg_weight] + [0.0] * (hardneg_causal_score.size(-1) - i - 1) for i in
                 range(hardneg_causal_score.size(-1))]
            ).to(device)
            # print("hard neg sample score:", weights)
            contrastive_causal_score = contrastive_causal_score + weights

        labels = torch.arange(contrastive_causal_score.size(0)).long().to(device)

        loss = self.contrastive_loss(contrastive_causal_score, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=contrastive_causal_score
        )

    def eval_forward(self, input_ids, attention_mask, labels, seg_ids=None, length=None):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        device = input_ids.device

        # Sentence pooler encoding
        pooler_output = self.forward_sent_encoding(input_ids, attention_mask, labels, seg_ids, length)  # [bs, num_sent, hidden_size]

        # Separate representation
        premise, hypothesis_0, hypothesis_1 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]

        # Scoring hypothesis 0
        causes_0, effects_0 = self.compose_causal_pair(premise, hypothesis_0, labels, device)
        premise_0_score_matrix = self.sim(causes_0, effects_0)
        score_0 = torch.diagonal(premise_0_score_matrix, offset=0).unsqueeze(1)

        # Scoring hypothesis 1
        causes_1, effects_1 = self.compose_causal_pair(premise, hypothesis_1, labels, device)
        premise_1_score_matrix = self.sim(causes_1, effects_1)
        score_1 = torch.diagonal(premise_1_score_matrix, offset=0).unsqueeze(1)

        # score = torch.cat([score_0, score_1], dim=1)
        return score_0, score_1