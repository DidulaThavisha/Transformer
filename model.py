import torch
import torch.nn as nn
import torch.nn.functional as F
from config import parse_option



opt = parse_option()

torch.manual_seed(opt.seed)
VOCAB_SIZE = opt.vocab_size
WINDOW_SIZE = opt.window_size
HEAD_SIZE = opt.head_size
N_EMBED = opt.n_embed
N_LAYERS = opt.n_layer
DEVICE = opt.device
N_HEADS = N_EMBED//HEAD_SIZE



class DecoderHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBED, HEAD_SIZE)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE)
        self.tril =  torch.tril(torch.ones((WINDOW_SIZE,WINDOW_SIZE), device=DEVICE)).to(DEVICE)

    def forward(self, x):
        x = x                                                                       # (B,T,C) C = 27
        k = self.key(x)                                                             # (B,T,C) C = 16
        q = self.query(x)                                                           # (B,T,C) C = 16 
        v = self.value(x)                                                           # (B,T,C) C = 16 
        weights = q @ k.transpose(1,2) * HEAD_SIZE**-0.5                                             # (B,T,C) @ (B,C,T) -->  (B,T,T)            T=8, C=16         
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


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = MultiHead()
        self.ffn = nn.Sequential(
            nn.Linear(N_EMBED,N_EMBED*4),
            nn.ReLU(),
            nn.Linear(4*N_EMBED,N_EMBED)
        )
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.decoder(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class TextGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.postional_embedding = nn.Embedding(WINDOW_SIZE, N_EMBED)
        self.decoder_block = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(N_EMBED)
        self.ln = nn.Linear(N_EMBED,VOCAB_SIZE)

    def forward(self, x):
        token_embed = self.token_embedding(x)
        pos_embed = self.postional_embedding(torch.arange(WINDOW_SIZE, device=DEVICE))
        x = token_embed + pos_embed
        x = self.decoder_block(x)
        x = self.norm(x)
        x = self.ln(x)
        return x


