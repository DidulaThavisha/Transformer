
import torch
import numpy as np
from model import TextGenerator
from dataset import TextDataset
from config import parse_option

opt = parse_option()


start = ['e','m','i','l','y','.','a','n']



def generate():
    instance = TextDataset(start)
    string = 'emily.an'
    emb = torch.tensor([instance.stoi[i] for i in string]).unsqueeze(0)
    model = TextGenerator()
    model.load_state_dict(torch.load(opt.load_path))
    model.eval()
    for _ in range(0, 100):
        with torch.no_grad(): 
            pred = model(emb)
            B, T, C = pred.shape
            emb = pred.argmax(dim=-1).view(B*T).unsqueeze(0)
            back_to_sting = "".join([instance.itos[ix] for ix in emb[0].cpu().numpy()])
            print(back_to_sting)

            

            

    # model = TextGenerator()
    # model.eval()
    # infer = []
    # for j in range(0, len(words_val)-window_size*batch_size,window_size*batch_size):
    #     x,y = instance[j]
    #     with torch.no_grad():            
    #         pred = model(x)
    #         B, T, C = pred.shape
    #         infer.append(pred.argmax(dim=-1).view(B*T).cpu().numpy())
    #         pred = pred.view(B*T, C) 
    # infer = np.array(infer).flatten()
    # string = "".join([instance.itos[ix] for ix in infer])
    # string = list(string.split('.'))
    # with open('generated.txt', 'w') as f:
    #     for item in string:
    #         if len(item)>1:
    #             f.write("%s\n" % item)

generate()