
import time

import torch
from data import get_data_large
from util import get_batch_data

torch.manual_seed(1)

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()
training_data = list(zip(citing_sentences, polarities))


for sentences, targets, seq_lengths in get_batch_data(training_data, 20, word_to_idx, shuffle=True):
    print(sentences[0], seq_lengths)
    time.sleep(50)
