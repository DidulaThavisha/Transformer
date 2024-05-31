import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from config import parse_option
from model import TextGenerator
from dataset import TextDataset

def train_supervised(epochs, lr, window_size, batch_size, words_train, words_val):
    min_val_loss = 100
    model_to_save = nn.Module
    criterion = nn.CrossEntropyLoss()
    model = TextGenerator()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for i in range (epochs):
        train_loss = 0
        instance = TextDataset(words_train)
        iterations = 0
        model.train()
        optimizer.zero_grad()
        for j in range(0, len(words_train)-window_size*batch_size,window_size*batch_size):
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
        instance = TextDataset(words_val)
        iterations = 0
        model.eval()
        infer = []
        for j in range(0, len(words_val)-window_size*batch_size,window_size*batch_size):
            x,y = instance[j]
            with torch.no_grad():            
                pred = model(x)
                B, T, C = pred.shape
                infer.append(pred.argmax(dim=-1).view(B*T).cpu().numpy())
                pred = pred.view(B*T, C)            
                y = y.view(B*T)
                loss = criterion(pred, y)
                val_loss += loss.item()
                iterations +=1
        val_loss = val_loss/iterations
        print(f"epoch:{i} ===>  train loss: {train_loss}  |   val loss: {val_loss}")
        if val_loss<min_val_loss:
            min_val_loss = val_loss
            model_to_save = model

    return model_to_save


def main():
    opt = parse_option()

    torch.manual_seed(opt.seed)
    words = open(opt.data_path, 'r').read().splitlines()
    random.shuffle(words)
    WORD_LIST = words
    WORD_LIST_TRAIN = WORD_LIST[:9*len(words)//10]
    WORD_LIST_VAL = WORD_LIST[9*len(words)//10:]

    words_train = ".".join(WORD_LIST_TRAIN)
    words_val = ".".join(WORD_LIST_VAL)
    window_size = opt.window_size
    batch_size = opt.batch_size
    lr = opt.learning_rate
    epochs = opt.epochs
    model = train_supervised(epochs, lr, window_size, batch_size, words_train, words_val)    
    torch.save(model.state_dict(), opt.save_path)


if __name__ == '__main__':
    main()
   