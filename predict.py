"""
Part of BME595 project
Program:
  Classify citation  models for citation classification
"""

import torch
import torch.autograd as autograd
from collections import Counter
from data import get_data_large
from util import prepare_sequence, get_batch_data

import time
model = torch.load('lstm-citation-classification.epoch3.ckpt')

test_citing_sentences, test_polarities, word_to_idx, polarity_to_idx = get_data_large(training=False)

# Just return an int label
def evaluate(sentence):
    inputs = prepare_sequence(sentence, word_to_idx)
    model.hidden = model.init_hidden()
    labels = model(inputs)
    return labels.data.max(1)[1][0]

def evaluate_batch(sentences, seq_lengths):
    sentences_in = autograd.Variable(sentences)
    model.hidden = model.init_hidden()
    labels = model(sentences_in, seq_lengths)
    return labels.data.max(1)[1]

def correct_rate():
    test_data = list(zip(test_citing_sentences, test_polarities))

    count = 0
    ctr_p = Counter()
    ctr_t = Counter()
    for sentences, targets, seq_lengths in get_batch_data(test_data, 20, word_to_idx, shuffle=True):
        predicts = evaluate_batch(sentences, seq_lengths)
        ctr_p += Counter(predicts)
        ctr_t += Counter(targets)
        count += (predicts == targets).sum()
    print(ctr_p, ctr_t)
    print('correct rate: ', count / len(test_data))


if __name__ == '__main__':
    correct_rate()
    # sentence = ['<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', 'table', 'tbl', 'stand', 'brill', 'transformationbased', 'errordriven', 'tagget', 'brill', '1995', 'stand', 'tagger', 'base', 'maimum', 'entropy', 'model', 'ratnaparkhi', '1996', 'spatter', 'stand', 'statistical', 'parser', 'base', 'decision', 'tree', 'magerman', '1996', 'igtree', 'stand', 'memorybased', 'tagger', 'daelemans', 'et', 'al']
    # for word in sentence:
    #     if word not in word_to_idx:
    #         print(word)