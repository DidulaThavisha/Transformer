import torch
import torch.nn as nn
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()
WORD_LIST = words
WORDS = ".".join(WORD_LIST)
VOCAB_SIZE = len(sorted((set(WORDS))))
WINDOW_SIZE = 8
HEAD_SIZE = 16



class TransformerDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE,VOCAB_SIZE)
        self.key = nn.Linear(VOCAB_SIZE, HEAD_SIZE)
        self.query = nn.Linear(VOCAB_SIZE, HEAD_SIZE)


    def forward(self, x):
        x = self.token_embedding(x)                 # (B,T,C) C = 27
        k = self.key(x)                             # (B,T,C) C = 16
        q = self.query(x)                           # (B,T,C) C = 16              
        weights = q @  k.transpose(1,2)             # (B,T,C) @ (B,C,T) -->  (B,T,T)            T=8, C=16
        tril =  torch.tril(torch.ones((WINDOW_SIZE,WINDOW_SIZE)))
        weights = weights.masked_fill(tril==0, float('-inf'))
        weights = F.softmax(weights,dim=1)
        print(weights)

        return weights

