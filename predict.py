"""
Part of BME595 project
Program:
  Classify citation  models for citation classification
"""

import torch
import torch.autograd as autograd
from collections import Counter
from data import get_data_large, get_combined_data
from util import prepare_sequence, get_batch_data, load_checkpoint

from constants import BATCH_SIZE

import time

torch.manual_seed(1)

test_citing_sentences, test_polarities, word_to_idx, polarity_to_idx = get_combined_data(training=False)

# Just return an int label
def evaluate(model, sentence):
    inputs = prepare_sequence(sentence, word_to_idx)
    model.hidden = model.init_hidden()
    labels = model(inputs)
    return labels.data.max(1)[1][0]

def evaluate_batch(model, sentences, seq_lengths):
    sentences_in = autograd.Variable(sentences)
    model.hidden = model.init_hidden()
    labels = model(sentences_in, seq_lengths)
    return labels.data.max(1)[1]

def get_error_rate(model=None, verbose=False):
    if not model:
        checkpoint = load_checkpoint()
        model = checkpoint['model']

    test_data = list(zip(test_citing_sentences, test_polarities))

    count = 0
    ctr_p = Counter()
    ctr_t = Counter()
    total_count = 0
    for sentences, targets, seq_lengths in get_batch_data(test_data, BATCH_SIZE, word_to_idx, shuffle=True):
        predicts = evaluate_batch(model, sentences, seq_lengths)
        ctr_p += Counter(predicts)
        ctr_t += Counter(targets)
        count += (predicts != targets).sum()
        total_count += BATCH_SIZE
    error_rate = count / total_count
    if verbose:
        print(ctr_p, ctr_t)
        print('error rate: ', error_rate)
    return error_rate


if __name__ == '__main__':
    get_error_rate(verbose=True)
    # sentence = ['<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', '<START>', 'table', 'tbl', 'stand', 'brill', 'transformationbased', 'errordriven', 'tagget', 'brill', '1995', 'stand', 'tagger', 'base', 'maimum', 'entropy', 'model', 'ratnaparkhi', '1996', 'spatter', 'stand', 'statistical', 'parser', 'base', 'decision', 'tree', 'magerman', '1996', 'igtree', 'stand', 'memorybased', 'tagger', 'daelemans', 'et', 'al']
    # for word in sentence:
    #     if word not in word_to_idx:
    #         print(word)