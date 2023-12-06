# todo
from torch.utils.data import DataLoader
import torch
from q1 import pad_and_convert_to_tensor
from data_rnn import load_imdb
from q2_3 import Model2
from q4 import Model4
import torch.nn as nn

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


def main():

    model_config = [Model2, Model4, Model4]
    # hyper parameters
    lr_config = [0.1, 0.01, 0.001, 0.0001]
    batch_size_config = [4, 8, 16, 32]

    # load data
    PAD_LEN = 100
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    # pad and convert to tensor
    x_train = pad_and_convert_to_tensor(x_train, PAD_LEN, w2i)
    x_val = pad_and_convert_to_tensor(x_val, PAD_LEN, w2i)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    # move data to device
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    x_val = x_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    
    vocab_size = len(w2i)
    emb_size = 300
    hidden_size = 300
    num_classes = numcls

    EPOCHS = 2

    results = {}

    # Grid search hyperparameter tuning:
    for i, model_class in enumerate(model_config):        
        for batch_size in batch_size_config:
            for lr in lr_config:
                print(f'tuning model {i}, batch_size = {batch_size}, lr = {lr}')
                # load one of the models
                if i == 1:
                    model = model_class(vocab_size, emb_size, hidden_size, num_classes, layer_type='lstm')
                elif i == 2:
                    model = model_class(vocab_size, emb_size, hidden_size, num_classes, layer_type='elman')
                else:
                    model = model_class(vocab_size, emb_size, hidden_size, num_classes)
                # load current batch size and learning rate
                BATCH_SIZE = batch_size

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                # move model to device
                model.to(DEVICE)
                
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

                results["model_"+str(i)+'_size_'+str(batch_size)+'_lr_'+str(lr)]

    with open('hyperparameter_tuning_results.txt', 'w') as file:
        for key, value in results.items():
            file.write(f'{key}: {value}\n')
    
    file.close()
#i = 0 - model_q2
#i = 1 - model_q4(lstm)
#i = 2 - model_q4(elman)

if __name__ == '__main__':
    main()
