import argparse
import torch

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, default='lstm', help='model name', required=False)
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size', required=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate', required=False)
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs', required=False)
    parser.add_argument('--window_size', type=int, default=8, help='window size', required=False)
    parser.add_argument('--head_size', type=int, default=4, help='Single head size of the MultiHeadAttention', required=False)
    parser.add_argument('--n_embed', type=int, default=16, help='Embedding size of the model', required=False)
    parser.add_argument('--n_layer',type=int, default=3, help='Number of Decoder Blocks', required=False)
    parser.add_argument('--seed', type=int, default=42, help='random seed', required=False)
    parser.add_argument('--save_path', type=str, default='models/model.pth', help='save path', required=False)
    parser.add_argument('--load_path', type=str, default='models/model.pth', help='load path', required=False)
    parser.add_argument('--data_path', type=str, default='names.txt', help='data path', required=False)
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer', required=False)
    parser.add_argument('--device', type=torch.device, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), help='device', required=False)
    opt = parser.parse_args()

    words = open(opt.data_path, 'r').read().splitlines()
    words = ".".join(words)
    opt.vocab = sorted((set(words)))
    opt.vocab_size = len(sorted((set(words))))

    return opt