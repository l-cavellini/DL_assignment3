"""
Epoch 8 Loss: 0.0023
Epoch 9 Loss: 0.0025
Epoch 10 Loss: 0.0011
Validation accuracy: 0.83
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class Model2(nn.Module):
    def __init__(
        self, vocab_size: int, emb_size: int, hidden_size: int, num_classes: int
    ):
        super(Model2, self).__init__()
        # layer 1
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # layer 2
        self.linear1 = nn.Linear(emb_size, hidden_size)
        # layer 5
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):  # x: (batch, time)
        # layer 1
        x = self.embedding(x)  # (batch, time, emb)
        # layer 3
        x = F.relu(self.linear1(x))  # (batch, time, hidden)
        # layer 4
        x, _ = torch.max(x, dim=1)  # Global max pooling (batch, hidden)
        return self.linear2(x)  # (batch, num_classes)

