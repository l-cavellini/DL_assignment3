import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from q1 import pad_and_convert_to_tensor
from data_rnn import load_imdb


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class SimpleSequenceModel(nn.Module):
    def __init__(
        self, vocab_size: int, emb_size: int, hidden_size: int, num_classes: int
    ):
        super(SimpleSequenceModel, self).__init__()
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


def main():
    EPOCHS = 10
    BATCH_SIZE = 32
    PAD_LEN = 100
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    # pad and convert to tensor
    x_train = pad_and_convert_to_tensor(x_train, PAD_LEN, w2i)
    x_val = pad_and_convert_to_tensor(x_val, PAD_LEN, w2i)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    vocab_size = len(w2i)
    emb_size = 300
    hidden_size = 300
    num_classes = numcls

    model = SimpleSequenceModel(vocab_size, emb_size, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # move model to device
    model.to(DEVICE)
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    # create data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # train
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")

    # move eval data to device
    x_val = x_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    # evaluate
    model.eval()
    # compute the accuracy on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in zip(x_val, y_val):
            batch = batch.unsqueeze(0).to(DEVICE)
            labels = labels.unsqueeze(0).to(DEVICE)
            output = model(batch)
            predicted = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Validation accuracy: {val_acc:.2f}")


if __name__ == "__main__":
    main()
