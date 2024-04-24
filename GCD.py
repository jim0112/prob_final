import os
import sys
import json
import numpy as np
import random
import nltk
import math

np.set_printoptions(threshold=np.inf)

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import setting
import time

class GCDDataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C
    
    def is_prime(self, x):
        if (x > 1):
            divisor = 2
            for i in range(divisor,x):
                if (x % i) == 0:
                    return False
        else:
            return False
        
        return True
    
    def gcd(self):
        data = []
        prime_data = [] # (a, b) = (a, 2*a)
        for a in range(1, 100):
            tmp = []
            for b in range(a+1, 100):
                c = math.gcd(a, b)
                res = f"{a:02}{b:02}{c:02}"

                if self.is_prime(a) and b % a == 0:
                    tmp.append([int(x) for x in res])
                else:
                    data.append([int(x) for x in res])
            if len(tmp) > 0:
                prime_data.append(tmp)
        return data, prime_data
    
    def __init__(self, config, split, seed):
        self.seed = seed
        self.config = config
        self.split = split # train/test

        self.config.ndigit = 2
        data, prime_data = self.gcd()

        test_data, train_data = [], []
        
        random.Random(seed).shuffle(data)
        for l in prime_data:
            random.Random(seed).shuffle(l)
            part = len(l) // 5
            test_data += l[:part]
            train_data += l[part:]

        num_test = min(int(len(data)*0.2), 500) - len(test_data) # 20% of the whole dataset, or only up to 500
        test_data = test_data + data[:num_test]
        train_data = train_data + data[num_test:]

        test_data = torch.tensor(test_data, dtype=torch.long)
        train_data = torch.tensor(train_data, dtype=torch.long)
        
        self.ixes = test_data if split == 'test' else train_data

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        return 2 * self.config.ndigit + 2 * (self.config.ndigit - 1) - 1

    def __len__(self):
        return self.ixes.size(0)

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        x = self.ixes[idx][:-1]
        y = self.ixes[idx][1:].clone() # predict the next token in the sequence
        y[:ndigit * 2 - 1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        return x, y
    
def eval_split(device, model, dataset):
    ndigit = dataset.config.ndigit

    loader = DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)
    total_correct = 0
    for _, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # isolate the first two digits of the input sequence alone
        d1d2 = x[:, :ndigit * 2]
        d3_gt = y[:, ndigit * 2 - 1:]

        d1d2d3 = model.generate(d1d2, ndigit, do_sample=False) # using greedy argmax, not sampling
        d3_pred = d1d2d3[:, ndigit * 2:]

        # evaluate the correctness of the results in this batch
        correct = torch.sum(torch.all(d3_pred == d3_gt, dim=1))
        total_correct += correct 
        # print(correct)
    return total_correct / len(dataset)