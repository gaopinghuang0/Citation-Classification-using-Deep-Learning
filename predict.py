"""
Part of BME595 project
Program:
  Classify citation  models for citation classification
"""

import torch
import time
from collections import Counter
from data import get_data_large
from util import prepare_sequence

model = torch.load('lstm-citation-classification.ckpt')

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()

# Just return an int label
def evaluate(sentence):
    inputs = prepare_sequence(sentence, word_to_idx)
    labels = model(inputs)
    return labels.data.max(1)[1][0]

def correct_rate():
    test_data = list(zip(citing_sentences, polarities))
    count = 0
    ctr_p = Counter()
    ctr_t = Counter()
    for sentence, target in test_data:
        predict = evaluate(sentence)
        ctr_p[predict] += 1
        ctr_t[target] += 1
        if predict == target:
            count += 1
    print(ctr_p, ctr_t)
    print('correct rate: ', count / len(test_data))


# # print(losses)
if __name__ == '__main__':
    correct_rate()