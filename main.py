import shutil
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


def save_model(config, model, config_path):
    """
    Save trained model as pt file with corresponding config
    :param config: configuration dict containing name of model
    :param model: trained pytorch model
    :param config_path: path to the current config file
    """

    model_base_path = os.path.join("trained_models/", config["model_save_name"])
    version_counter = 1
    model_save_path = f"{model_base_path}_{version_counter}.pt"

    while os.path.exists(model_save_path):
        version_counter += 1
        model_save_path = f"{model_base_path}_{version_counter}.pt"

    config_save_path = f"{model_base_path}_{version_counter}_config.yaml"

    print(f"Saving model to {model_save_path} and config to {config_save_path}")
    torch.save(model.state_dict(), model_save_path)
    shutil.copyfile(config_path, config_save_path)


def load_model(config, vocab_size):
    """
    Load trained model from pt file and init it with corresponding config
    :param config: configuration dict containing name of model to load
    :param vocab_size: size of the vocabulary
    """

    model_load_path = os.path.join("trained_models/", config["model_load_name"] + ".pt")
    print(f"Loading model from {model_load_path}")

    if not os.path.exists(model_load_path):
        raise FileNotFoundError(f"Model {model_load_path} not found.")

    config_load_path = os.path.join("trained_models/", config["model_load_name"] + "_config.yaml")
    if not os.path.exists(config_load_path):
        raise FileNotFoundError(f"Config {config_load_path} not found.")

    with open(config_load_path, 'r') as config_file:
        saved_config = yaml.safe_load(config_file)

    model = VanillaGPT(vocab_size, saved_config["context_size"], saved_config["d_model"], saved_config["num_heads"],
                       saved_config["num_layers"], saved_config["dropout"], saved_config["do_use_rotary_emb"], device)

    state_dict = torch.load(str(model_load_path), weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def main():

    config_name = sys.argv[1]
    config_path = os.path.join("configs/", config_name)
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    optimizer = get_optimizer(config)
    loss_function = get_loss_function(config)

    train_data, val_data, vocab_size = load_dataset()

    if config["load_model"] is True:
        model = load_model(config, vocab_size)
    else:
        model = VanillaGPT(vocab_size, config["context_size"], config["d_model"], config["num_heads"], config["num_layers"], config["dropout"], config["do_use_rotary_emb"], device)
        model = model.to(device)

        tr = Trainer(model, train_data, val_data, vocab_size, config["context_size"], config["batch_size"], config["train_iters"], config["eval_iters"], loss_function, optimizer, config["learning_rate"], device)
        tr.train()

        if config["save_model"] is True:
            save_model(config, model, config_path)

    # Test generator
    context = torch.zeros((1, 1), dtype=torch.long).to(device)
    # gen_out = model.generate(context, config["context_size"], num_tokens_to_generate=2000)[0].tolist()
    gen_out = model.generate_with_kv_cache(context, config["context_size"],  num_tokens_to_generate=2000)[0].tolist()

    print(decode_chars(gen_out, itos_lut))


if __name__ == '__main__':
    main()