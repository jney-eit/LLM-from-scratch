import subprocess
import os
import sys
from pickletools import optimize

import yaml
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


from models import VanillaGPT
from trainer import Trainer

stoi_lut = {}
itos_lut = {}



def gen_encode_lut(chars):
    """
    Creates LUT that maps all unique chars to a specific integer value
    :param chars: list of unique chars
    :return: dict with chars as key and integer as value
    """
    stoi = {ch:i for i,ch in enumerate(chars)}
    return stoi

def gen_decode_lut(chars):
    """
    Creates lut that maps integer value of each char back to char
    :param chars: list of unique chars
    :return: dict with integer as key and char as value
    """
    itos = {i:ch for i,ch in enumerate(chars)}
    return itos


def encode_chars(text, stoi):
    """
    Encodes text using lut
    :param text: text to encode
    :param stoi: char to int lut
    :return: encoded text
    """
    return [stoi[c] for c in text]

def decode_chars(l, itos):
    """
    Decodes text using lut
    :param l: encoded text
    :param itos: int to char lut
    :return: decoded text
    """
    return ''.join([itos[i] for i in l])


def load_dataset(train_test_ratio=0.9):
    """
    Loads and processes tinyshakespeare dataset
    :return: training dataset, validation dataset and size of the vocabulary
    """

    # load dataset
    if not os.path.isfile("input.txt"):
        subprocess.call("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", shell=True)

    # read dataset
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Length of the dataset in characters: {len(text)}")

    # get all unique chars in dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print(f"All chars: {''.join(chars)}")
    print(f"Vocab size: {vocab_size}")

    # global look-up tables
    global stoi_lut
    global itos_lut

    # generate LUTs
    stoi_lut = gen_encode_lut(chars)
    itos_lut = gen_decode_lut(chars)

    # encode entire dataset
    data = torch.tensor(encode_chars(text, stoi_lut), dtype=torch.long)

    # split dataset
    num_train_chars = int(train_test_ratio * len(data))
    train_data = data[:num_train_chars]
    val_data = data[num_train_chars:]

    return train_data, val_data, vocab_size



def print_data(x, y):
    """
    Print input and target sample
    :param x: inputs
    :param y: targets
    """

    print("inputs:")
    print(x.shape)
    print(x)

    print("targets:")
    print(y.shape)
    print(y)

    print("---------------------------")

    for b in range(x.shape[0]): # batch dimension
        for t in range(x.shape[1]): # time dimension
            context = x[b, :t+1]
            target = y[b, t]
            print(f"when input is {context.tolist()}, the target is {target}")



def get_optimizer(config):
    """
    Get optimizer from config
    :param config: config dict
    :return: optimizer
    """
    if config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")

    return optimizer


def get_loss_function(config):
    """
    Get optimizer from config
    :param config: config dict
    :return: loss function
    """

    if config["loss_function"] == "CrossEntropy":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function {config['loss_function']} not implemented")

    return loss_function



def main():

    config_path = os.path.join("configs/", sys.argv[1])
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    optimizer = get_optimizer(config)
    loss_function = get_loss_function(config)

    train_data, val_data, vocab_size = load_dataset()
    model = VanillaGPT(vocab_size, config["context_size"], config["d_model"], config["num_heads"], config["num_layers"], config["dropout"], device)
    model = model.to(device)

    tr = Trainer(model, train_data, val_data, vocab_size, config["context_size"], config["batch_size"], config["train_iters"], config["eval_iters"], loss_function, optimizer, config["learning_rate"], device)
    tr.train()

    # Test generator
    context = torch.zeros((1, 1), dtype=torch.long).to(device)
    print(decode_chars(model.generate(context, config["context_size"], num_tokens_to_generate=2000)[0].tolist(), itos_lut))


if __name__ == '__main__':
    main()