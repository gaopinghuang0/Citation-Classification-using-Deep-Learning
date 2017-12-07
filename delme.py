
import time

import torch
from data import data_loader
from util import get_batch_data

torch.manual_seed(1)

data, word_to_idx, label_to_idx = data_loader(training=True)
print(len(data))

data, word_to_idx, label_to_idx = data_loader(training=False)
print(len(data))
