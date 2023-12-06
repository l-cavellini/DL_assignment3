import torch
from data_rnn import load_imdb

def pad_and_convert_to_tensor(sequences: list, max_len: int, w2i: dict) -> torch.Tensor:
    pad_index = w2i['.pad']
    # initialize a matrix of shape (len(sequences), max_len) with the pad_value
    padded_sequences = torch.full((len(sequences), max_len), pad_index, dtype=torch.long)

    # for every sequence
    for i, sequence in enumerate(sequences):
        # get the length of the sequence
        length = min(max_len, len(sequence))
        # replace the values in the i-th row of padded_sequences with the sequence
        padded_sequences[i, :length] = torch.tensor(sequence[:length], dtype=torch.long)

    return padded_sequences



def main():
    MAX_LEN = 10
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
    batch_tensor = pad_and_convert_to_tensor(x_train, MAX_LEN, w2i)
    print(batch_tensor)


if __name__ == '__main__':
    main()

