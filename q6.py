import torch.nn as nn
import torch


class AutoRegressive_lstm(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_layers):
        super(AutoRegressive_lstm, self).__init__()
        # layer 1
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # layer 2: LSTM layer
        self.lstm = nn.LSTM(
            emb_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        # layer 3: output layer
        self.linear3 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor):
        # layer 1
        x = self.embedding(x)
        # layer 2
        x, _ = self.lstm(x)
        # layer 3 (output)
        x = self.linear3(x)
        return x
