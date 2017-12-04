"""
Part of BME595 project
Program:
  Classify citation  models for citation classification
"""

import time  # debug
from collections import Counter

import torch
import torch.autograd as autograd

from sklearn import metrics
from data import get_combined_data
from util import prepare_sequence, get_batch_data, load_checkpoint

from constants import BATCH_SIZE


torch.manual_seed(1)

def evaluate(model, sentence, word_to_idx):
    """Return an int as the predict."""
    inputs = prepare_sequence(sentence, word_to_idx)
    model.hidden = model.init_hidden()
    labels = model(inputs)
    return labels.data.max(1)[1][0]

def evaluate_batch(model, sentences, seq_lengths):
    """
    Return a FloatTensor as predicts for a batch of sentences.
    """
    sentences_in = autograd.Variable(sentences)
    model.hidden = model.init_hidden()
    labels = model(sentences_in, seq_lengths)
    return labels.data.max(1)[1]

def get_error_rate(model=None, training=False, report=False):
    """
    Compute the overall error rate of the trained model.

    If training is False, use test_data, otherwise training_data.
    If report is True, print precision, recall, F1-score, and confusion matrix.
    """
    model = model or load_checkpoint()['model']

    sentences, polarities, word_to_idx, _ = get_combined_data(training)
    data = list(zip(sentences, polarities))

    targets = torch.LongTensor()
    predicts = torch.LongTensor()
    for sentences, _targets, seq_lengths in get_batch_data(
            data, BATCH_SIZE, word_to_idx, shuffle=True):

        _predicts = evaluate_batch(model, sentences, seq_lengths)
        targets = torch.cat((targets, _targets), 0)
        predicts = torch.cat((predicts, _predicts), 0)

    error_rate = (targets != predicts).sum() / targets.size(0)

    if report:
        print(Counter(targets.numpy()), Counter(predicts.numpy()))
        print('error rate: ', error_rate)
        labels = ('neutral', 'positive', 'negative')
        print('Report:\n', metrics.classification_report(
            targets.numpy(), predicts.numpy(), target_names=labels))
        print('Confusion matrix: \n', metrics.confusion_matrix(targets.numpy(), predicts.numpy()))

    return error_rate



if __name__ == '__main__':
    get_error_rate(training=False, report=True)
