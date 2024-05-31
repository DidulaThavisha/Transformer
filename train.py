import torch
from model import TextGenerator
from dataset import TextDataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random


torch.manual_seed(42)

words = open('names.txt', 'r').read().splitlines()
random.shuffle(words)
WORD_LIST = words
WORD_LIST_TRAIN = WORD_LIST[:9*len(words)//10]
WORD_LIST_VAL = WORD_LIST[9*len(words)//10:]

WORDS = ".".join(WORD_LIST_TRAIN)
WORDS_VAL = ".".join(WORD_LIST_VAL)
VOCAB_SIZE = len(sorted((set(WORDS))))
WINDOW_SIZE = 8
BATCH_SIZE = 16

criterion = nn.CrossEntropyLoss()
model = TextGenerator()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

for name, param in model.named_parameters():
    print(f'{name}: {param.requires_grad}')

for i in range (10):
    train_loss = 0
    instance = TextDataset(WORDS)
    iterations = 0
    model.train()
    optimizer.zero_grad()
    for j in range(0, len(WORDS)-WINDOW_SIZE*BATCH_SIZE,WINDOW_SIZE*BATCH_SIZE):
        x,y = instance[j]
        with torch.set_grad_enabled(True):
            pred = model(x)
            B, T, C = pred.shape
            pred = pred.view(B*T, C)
            y = y.view(B*T)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            iterations +=1    
    train_loss = train_loss/iterations

    val_loss = 0
    instance = TextDataset(WORDS_VAL)
    iterations = 0
    model.eval()
    for j in range(0, len(WORDS_VAL)-WINDOW_SIZE*BATCH_SIZE,WINDOW_SIZE*BATCH_SIZE):
        x,y = instance[j]
        with torch.no_grad():            
            pred = model(x)
            B, T, C = pred.shape
            pred = pred.view(B*T, C)
            y = y.view(B*T)
            loss = criterion(pred, y)
            val_loss += loss.item()
            iterations +=1
    val_loss = val_loss/iterations
    print("train loss: ",train_loss,'  |  ', "val loss: ", val_loss)

    