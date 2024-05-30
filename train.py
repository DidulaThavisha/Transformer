import torch
from model import TextGenerator
from dataset import TextDataset
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

words = open('names.txt', 'r').read().splitlines()
WORD_LIST = words
WORDS = ".".join(WORD_LIST)
VOCAB_SIZE = len(sorted((set(WORDS))))

criterion = nn.CrossEntropyLoss()
model = TextGenerator()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
model.train()
for name, param in model.named_parameters():
    print(f'{name}: {param.requires_grad}')
for i in range (1):
    instance = TextDataset(words)
    optimizer.zero_grad()
    for j in range(100):
        x,y = instance['train']
        pred = model(x)
        B, T, C = pred.shape
        pred = pred.view(B*T, C)
        y = y.view(B*T)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print(loss.item())