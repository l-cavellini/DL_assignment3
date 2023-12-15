import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from q6 import AutoRegressive_lstm
from data_rnn import load_ndfa, load_brackets
from torch.utils.data import DataLoader

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 3
BATCH_SIZE = 32
MAX_LEN = 20
# x_train, (i2w, w2i) = load_ndfa(n=150_000)
x_train, (i2w, w2i) = load_brackets(n=150_000)


def pad_and_convert_to_tensor_part4(
    sequences: list, max_len: int, w2i: dict
) -> torch.Tensor:
    pad_index = w2i[".pad"]
    start_token = w2i[".start"]
    end_token = w2i[".end"]
    # initialize a matrix of shape (len(sequences), max_len) with the pad_value
    padded_sequences = torch.full(
        (len(sequences), max_len), pad_index, dtype=torch.long
    )
    # add start and end tokens to the sequences
    for i in padded_sequences:
        i[0] = start_token
        i[-1] = end_token

    # for every sequence
    for i, sequence in enumerate(sequences):
        # get the length of the sequence
        length = min(max_len - 2, len(sequence))

        padded_sequences[i, 1 : length + 1] = torch.tensor(
            sequence[:length], dtype=torch.long
        )

    return padded_sequences


def convert_to_target(padded_sequences, max_len):
    # create a clone of the entire padded_sequences tensor
    target_sequences = padded_sequences.clone()

    for i in range(len(target_sequences)):
        # shift tokens to the left by 1 position for each sequence
        target_sequences[i, :-1] = padded_sequences[i, 1:]
        # append zero at the end for each sequence
        target_sequences[i, -1] = 0

    return target_sequences


# pad and convert to tensor
X_train_padded = pad_and_convert_to_tensor_part4(x_train, MAX_LEN, w2i)
y_train = convert_to_target(X_train_padded, MAX_LEN)

vocab_size = len(w2i)
emb_size = 32
hidden_size = 16

model = AutoRegressive_lstm(vocab_size, emb_size, hidden_size, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# move to device
model.to(DEVICE)
X_train_padded.to(DEVICE)
y_train.to(DEVICE)

# create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train_padded, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

writer = SummaryWriter()

criterion = nn.CrossEntropyLoss(reduction="sum")

optimizer = optim.Adam(model.parameters(), lr=0.001)

clip_value = 1

for epoch in range(50):  # Assuming 50 epochs
    total_loss = 0
    total_tokens = 0

    for (
        inputs,
        targets,
    ) in train_loader:  # Assuming your data loader is named train_loader
        optimizer.zero_grad()
        # move to device
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        # Forward pass
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])  # Reshape for cross-entropy
        targets = targets.view(-1)

        # Compute loss (ignore padding tokens if using masking)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # Update total loss and total tokens (adjust for your data)
        total_loss += loss.item()
        total_tokens += targets.size(0)

    avg_loss = total_loss / total_tokens
    writer.add_scalar("Loss/train", avg_loss, epoch)

    print(f"Epoch [{epoch+1}/{50}], Loss: {avg_loss:.4f}")

writer.close()
