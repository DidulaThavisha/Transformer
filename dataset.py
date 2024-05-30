import torch
from torch.utils.data import Dataset

# from train import VOCAB_SIZE, WORDS


class TextDataset(Dataset):
    def __init__(self, words):
        self.word_list = words
        self.words = ".".join(self.word_list)
        self.vocab = sorted((set(self.words)))
        self.stoi = {self.vocab[i]: i for i in range(27)}
        self.itos = {i: self.vocab[i] for i in range(27)}
        self.window_size = 8
        self.batch_size = 4

    def __getitem__(self, split):
        x = torch.zeros((self.batch_size, self.window_size))
        y = torch.zeros((self.batch_size, self.window_size))
        b = torch.randint(low=0, high=len(self.words), size=(self.batch_size,))
        for j in range(self.batch_size):
            x[j] = torch.tensor(
                [self.stoi[self.words[i]] for i in range(b[j], b[j] + self.window_size)]
            )
            y[j] = torch.tensor(
                [self.stoi[self.words[i]] for i in range(b[j] + 1, b[j] + self.window_size + 1)]
            )

        return x.to(torch.int64), y.to(torch.int64)
