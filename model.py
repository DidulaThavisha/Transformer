import torch
import torch.nn as nn
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()
WORD_LIST = words
WORDS = ".".join(WORD_LIST)
VOCAB_SIZE = len(sorted((set(WORDS))))
WINDOW_SIZE = 8
HEAD_SIZE = 4
N_EMBED = 16
N_HEADS = N_EMBED//HEAD_SIZE



class DecoderHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBED, HEAD_SIZE)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE)
        self.tril =  torch.tril(torch.ones((WINDOW_SIZE,WINDOW_SIZE)))

    def forward(self, x):
        x = x                                                                       # (B,T,C) C = 27
        k = self.key(x)                                                             # (B,T,C) C = 16
        q = self.query(x)                                                           # (B,T,C) C = 16 
        v = self.value(x)                                                           # (B,T,C) C = 16 
        weights = q @ k.transpose(1,2)                                              # (B,T,C) @ (B,C,T) -->  (B,T,T)            T=8, C=16         
        weights = weights.masked_fill(self.tril==0, float('-inf'))
        weights = F.softmax(weights,dim=1)
        x =  weights @ v                                                            # (B,T,T) @ (B,T,C) -->  (B,T,C)            T=8, C=16
        return x


class MultiHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList([DecoderHead() for _ in range(N_HEADS)])
    
    def forward(self, x):
        x = torch.cat([i(x) for i in self.heads], dim=-1)
        return x


class TextGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.postional_embedding = nn.Embedding(WINDOW_SIZE, N_EMBED)
        self.decoder = MultiHead()
        self.ffn = nn.Sequential(
            nn.Linear(N_EMBED,N_EMBED),
            nn.ReLU()
        )
        self.ln = nn.Linear(N_EMBED,VOCAB_SIZE)


    def forward(self, x):
        token_embed = self.token_embedding(x)
        pos_embed = self.postional_embedding(torch.arange(WINDOW_SIZE))
        x = token_embed + pos_embed
        x = x + self.decoder(x)
        x = x + self.ffn(x)
        x = self.ln(x)

        return x


