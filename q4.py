import torch
import torch.nn as nn

class Elman(nn.Module):
    def __init__(self, insize=300, outsize=300, hsize=300):
        super().__init__()
        self.hsize = hsize
        # Linear layer for input to hidden transformation
        self.lin1 = nn.Linear(insize + hsize, hsize)
        # Linear layer for hidden to output transformation
        self.lin2 = nn.Linear(hsize, outsize)

    def forward(self, input_seq, hidden=None):
        batch_size, seq_length, emb_size = input_seq.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hsize, dtype=torch.float, device=input_seq.device)

        outputs = []
        for i in range(seq_length):
            # Concatenating the input and hidden state
            combined_input = torch.cat([input_seq[:, i, :], hidden], dim=1)
            hidden = torch.tanh(self.lin1(combined_input))
            output = self.lin2(hidden)
            outputs.append(output[:, None, :])

        return torch.cat(outputs, dim=1), hidden


# nd build a second model, like the one in q2_3.py, but replacing the second layer with an Elman(300, 300, 300) layer
class Model4(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_classes: int):
        super(Model4, self).__init__()
        # layer 1
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # layer 2
        self.elman = Elman(emb_size, hidden_size, hidden_size)
        # layer 5
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):  # x: (batch, time)
        # layer 1
        x = self.embedding(x)  # (batch, time, emb)
        # layer 3
        x, _ = self.elman(x)  # (batch, time, hidden)
        # layer 4
        x, _ = torch.max(x, dim=1)  # Global max pooling (batch, hidden)
        return self.linear2(x)  # (batch, num_classes)

