import os
import sys
import json
import numpy as np
import random
import nltk
np.set_printoptions(threshold=np.inf)

import torch
from ChickenRabbit import ChickenRabbitDataset, batch_end_callback
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.model_multiplier import GPT
from mingpt.trainer_multiplier import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from itertools import permutations

# -----------------------------------------------------------------------------

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 5487
    C.system.work_dir = './out/ChickenRabbit' # can change to any

    # data
    C.data = ChickenRabbitDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    
    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 1e-3 # the model we're using is so small that we can go a bit faster
    C.trainer.task = "ChickenRabbit"
    return C

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    config = get_config()
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = ChickenRabbitDataset(config.data, split='train', seed=0)
    test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=0)

    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    model = model.to(device)

    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()


    
