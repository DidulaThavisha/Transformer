import torch
from torch.utils.data import Dataset
from config import parse_option

opt = parse_option()

class TextDataset(Dataset):
    def __init__(self, words):        
        self.words = words
        self.vocab = opt.vocab
        self.vocab_size = opt.vocab_size
        self.stoi = {self.vocab[i]: i for i in range(self.vocab_size)}
        self.itos = {i: self.vocab[i] for i in range(self.vocab_size)}
        self.window_size = opt.window_size
        self.batch_size = opt.batch_size

    def __getitem__(self, ix):
        x = torch.zeros((self.batch_size, self.window_size))
        y = torch.zeros((self.batch_size, self.window_size))
        b = torch.tensor([ix+i*self.window_size for i in range(self.batch_size)])
        for j in range(self.batch_size):
            x[j] = torch.tensor(
                [self.stoi[self.words[i]] for i in range(b[j], b[j] + self.window_size)]
            )
            y[j] = torch.tensor(
                [self.stoi[self.words[i]] for i in range(b[j] + 1, b[j] + self.window_size + 1)]
            )

        return x.to(torch.int64), y.to(torch.int64)
    
    def itos(self, ix):
        return self.itos[ix]
    
    def stoi(self, string):
        return self.stoi[string]
