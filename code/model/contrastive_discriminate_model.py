from turtle import forward
import torch.nn as nn
import torch
from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
from transformers import OpenAIGPTConfig, OpenAIGPTModel, XLNetConfig, XLNetModel
from transformers import BartConfig, BartForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from .discriminate_model import pretrained_model


class CosSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
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
        pair = torch.cat([x, y], dim=-1)
        score = self.score_mlp(pair).squeeze()
        return score / self.temp
    
    def init_weights(self):
        for m in self.score_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)


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

        self.cause_emb = MLPLayer(self.config)
        self.effect_emb = MLPLayer(self.config)

        # embedding of the causal-effect pair
        # init mlp weights
        # self.causal_mlp = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        if hps.score == "cossim":
            self.sim = CosSimilarity()
        elif hps.score == "causalscore":
            self.sim = Scorer(self.config)

        # by default, the loss func is "BCE"
        self.contrastive_loss = nn.CrossEntropyLoss()

    # compose causal pairs
    def compose_causal_pair(self, premise, hypothesis, labels):
        """
        Arguments:
            premise     [batch_size, hidden_size]
            hypothesis  [batch_size, hidden_size]
            labels     [batch_size, 3]
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
        return causes, effects

    def forward(self, input_ids, attention_mask, labels, seg_ids=None, length=None, mode='train'):
        if mode == 'train':
            return self.cl_forward(input_ids, attention_mask, labels, seg_ids, length)
        else:
            return self.eval_forward(input_ids, attention_mask, labels, seg_ids, length)

    def cl_forward(self, input_ids, attention_mask, labels, seg_ids=None, length=None):

        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if seg_ids is not None:
            seg_ids = seg_ids.view((-1, seg_ids.size(-1)))
        if length is not None:
            length = length.view(-1)
        # labels = labels.view(-1)
        # print("input_ids.size:", input_ids.size())
        # print("attention_mask.size:", attention_mask.size())
        # print("labels.size:", labels.size())
        # print("seg_ids.size:", seg_ids.size())
        # print("length.size:", length.size())

        # print("input device:", input_ids.device)

        sent_embs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=seg_ids)

        # Pooling
        # by default, use the "cls" embedding as the sentence representation
        pooler_output = sent_embs.pooler_output
        # print("pooler_output.size:", pooler_output.size())
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # Separate representation
        premise, hypothesis_0 = pooler_output[:, 0], pooler_output[:, 1]
        # print("premise.size:", premise.size())
        # print("hypothesis.size:", hypothesis_0.size())
        causes_0, effects_0 = self.compose_causal_pair(premise, hypothesis_0, labels)
        causes_0 = causes_0.to(input_ids.device)
        effects_0 = effects_0.to(input_ids.device)
        causes_0 = self.cause_emb(causes_0)
        effects_0 = self.effect_emb(effects_0)
        contrastive_causal_score = self.sim(causes_0, effects_0)
        # print("contrastive score:", contrastive_causal_score.size())

        # Hard negative
        if num_sent == 3:
            hypothesis_1 = pooler_output[:, 2]
            causes_1, effects_1 = self.compose_causal_pair(premise, hypothesis_1, labels)
            causes_1 = causes_1.to(input_ids.device)
            effects_1 = effects_1.to(input_ids.device)
            causes_1 = self.cause_emb(causes_1)
            effects_1 = self.effect_emb(effects_1)
            hardneg_causal_score = self.sim(causes_1, effects_1)

            # Calculate loss with hard negatives
            # Note that weights are actually logits of weights

            # TODO: what is hard negative weight
            hard_neg_weight = self.hps.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (contrastive_causal_score.size(-1) - hardneg_causal_score.size(-1)) + [0.0] * i + [
                    hard_neg_weight] + [0.0] * (hardneg_causal_score.size(-1) - i - 1) for i in
                 range(hardneg_causal_score.size(-1))]
            ).to(input_ids.device)
            # print("hard neg sample score:", weights)
            contrastive_causal_score = contrastive_causal_score + weights

        # Hypothesis 0 is always the correct one (positive samples)
        # Hypothesis 1 is always the wrong one (negative samples)
        labels = torch.arange(contrastive_causal_score.size(0)).long().to(input_ids.device)
        # print("for loss, contrastive_causal_score:", contrastive_causal_score.size())
        # print("for loss, labels:", labels.size())
        # for i in range(batch_size):
        #     print(f"=== data {i} ===")
        #     print("score:", contrastive_causal_score[i])
        #     print("label:", labels[i])

        # Arg: contrastive_causal_score [batch_size, batch_size]
        # Arg: labels [batch_size]
        loss = self.contrastive_loss(contrastive_causal_score, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=contrastive_causal_score
        )

    def eval_forward(self, input_ids, attention_mask, labels, seg_ids=None, length=None):

        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if seg_ids is not None:
            seg_ids = seg_ids.view((-1, seg_ids.size(-1)))
        if length is not None:
            length = length.view(-1)

        sent_embs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=seg_ids)

        # Pooling
        # by default, use the "cls" embedding as the sentence representation
        pooler_output = sent_embs.pooler_output
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # Separate representation
        premise, hypothesis_0, hypothesis_1 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]

        causes_0, effects_0 = self.compose_causal_pair(premise, hypothesis_0, labels)
        causes_0 = causes_0.to(input_ids.device)
        effects_0 = effects_0.to(input_ids.device)
        causes_0 = self.cause_emb(causes_0)
        effects_0 = self.effect_emb(effects_0)
        score_0 = self.sim(causes_0, effects_0)

        causes_1, effects_1 = self.compose_causal_pair(premise, hypothesis_1, labels)
        causes_1 = causes_1.to(input_ids.device)
        effects_1 = effects_1.to(input_ids.device)
        causes_1 = self.cause_emb(causes_1)
        effects_1 = self.effect_emb(effects_1)
        score_1 = self.sim(causes_1, effects_1)

        # print("score_0:", score_0)
        # print("score_1:", score_1)
        return score_0, score_1