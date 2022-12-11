import torch.nn as nn


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


class CausalScorer(nn.Module):
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
