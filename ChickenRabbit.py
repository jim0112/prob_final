import os
import sys
import json
import numpy as np
import random
import nltk
np.set_printoptions(threshold=np.inf)

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.utils import set_seed, setup_logging, CfgNode as CN

class ChickenRabbitDataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 3
        return C
    
    def chicken_rabbit(self):
        """
        we only want valid number of chicks or rabbits, so we will first get 
        all the possible combination of c & r and calculate the corresponding 
        heads and legs.
        example data: [0, 0, 2, 0, 0, 5, 0, 1, 0, 4]
        target      : [0, 2, 0, 0, 5, 0, 1, 0, 4, x]
        expect      : [N, N, N, N, N, 0, 1, 0, 4, x]
        """
        data = []
        for i in range(100):
            for j in range(100):
                d = f"{i+j:03}{2*i+4*j:03}{i:02}{j:02}"
                data.append([int(dd) for dd in d])
        return data

    def __init__(self, config, split, seed):
        self.config = config
        self.split = split # train / test
        # split up all addition problems into either training data or test data
        ndigit = self.config.ndigit
        print(ndigit)
        #assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        data = self.chicken_rabbit()
        # print(data)
        print(len(data))

        random.Random(seed).shuffle(data)
        perm = torch.tensor(data, dtype=torch.long)

        num_test = min(int(len(perm)*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        # print(self.ixes.size(0))

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,axb, and 2*digits due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 2 * self.config.ndigit + 2 * (self.config.ndigit - 1) - 1

    def __len__(self):
        return self.ixes.size(0)

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # x will be input to GPT and y will be the associated expected outputs
        x = self.ixes[idx][:-1]
        y = self.ixes[idx][1:].clone() # predict the next token in the sequence
        y[:ndigit * 2 - 1] = -1 # we will only train in the output locations. -1 will mask loss to zero

        return x, y

def eval_split(device, model, dataset):
    ndigit = dataset.config.ndigit
    total_correct = 0
    loader = DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)
            
    for b, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # isolate the first two digits of the input sequence alone
        d1d2 = x[:, :ndigit * 2]
        d3_gt = y[:, ndigit * 2 - 1:]
        # print(f'the shape of d3_gt = {d3_gt.shape}')
        d1d2d3 = model.generate(d1d2, 2 * (ndigit - 1), do_sample=False) # using greedy argmax, not sampling
        d3_pred = d1d2d3[:, ndigit * 2:]
        # print(f'the shape of d3_pred = {d3_pred.shape}')

        # evaluate the correctness of the results in this batch
        correct = torch.sum(torch.eq(torch.sum(d3_pred == d3_gt, dim=1), ndigit * 2 - 2)).item()
        total_correct += correct 

    return total_correct, total_correct / len(dataset)

top_score = 0
counter = 0
max_train = 0

def batch_end_callback(trainer, model, train_dataset, test_dataset):
    global top_score
    global counter 
    global max_train
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    if trainer.iter_num % 50 == 0:
        # evaluate both the train and test score
        model.eval()
        with torch.no_grad():
            train_score, train_mean = eval_split(trainer.device, model, train_dataset)
            test_score, test_mean  = eval_split(trainer.device, model, test_dataset)
        print(f'the mean of train and test are {train_mean}, {test_mean}')
        score = train_score + test_score
        # save the model if this is the best score we've seen so far
        if score > top_score:
            top_score = score
            print(f"saving model with new top score of {score}")
            ckpt_path = os.path.join(f"out/{trainer.config.task}", "model_best.pt")
            torch.save(model.state_dict(), ckpt_path)
        # revert model to training mode
        if train_mean > max_train:
            max_train = train_mean
            counter = 0
        if train_mean < max_train:
            counter += 1
            print(counter)
        
        if counter >= 2000:
            trainer.config.max_iters = 0
        # revert model to training mode
        model.train()