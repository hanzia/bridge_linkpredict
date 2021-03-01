#python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import logging

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triples_set = set(triples)
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        position