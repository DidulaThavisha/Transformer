import torch
from model import TransformerDecoder
from dataset import TextDataset

torch.manual_seed(42)

words = open('names.txt', 'r').read().splitlines()
WORD_LIST = words
WORDS = ".".join(WORD_LIST)
VOCAB_SIZE = len(sorted((set(WORDS))))


for i in range (1):
    instance = TextDataset(words)
    model = TransformerDecoder()
    for j in range(1):
        x,y = instance['train']
        pred = model(x)
        print(pred.shape)
