"""
Part of BME595 project
Program:
  Train models for citation classification
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import time
import random

from model import BatchLSTM
from data import get_data_large, get_combined_data
from util import prepare_sequence, label_to_text, get_batch_data, \
                save_to_pickle, load_checkpoint, save_checkpoint
from predict import get_error_rate
from constants import *

torch.manual_seed(1)

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_combined_data()

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
RESUME = False
use_glove = True

print('total epochs: ', EPOCHS, '; use_glove: ', use_glove)

if RESUME:
    # load checkpoint
    checkpoint = load_checkpoint()
    model = checkpoint['model']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model...')
    model = BatchLSTM(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, len(word_to_idx), len(polarity_to_idx))
    if use_glove:
        model.load_glove_model('GloVe-1.2/vectors.txt', word_to_idx)

losses = []
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optim below are not working
# optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
# optimizer = optim.Adagrad(model.parameters(), lr=0.001)


# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# inputs = prepare_sequence(citing_sentences[0], word_to_idx)
# print(inputs)
# labels = model(inputs)
# print(labels)
# label_to_text(labels, polarity_to_idx)
since = time.time()
training_data = list(zip(citing_sentences, polarities))
training_error_rates = []
test_error_rates = []
for epoch in range(1, EPOCHS+1):
    total_loss = torch.Tensor([0])
    error_count = 0
    total_count = 0
    for sentences, targets, seq_lengths in get_batch_data(training_data, BATCH_SIZE, word_to_idx, shuffle=True):
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        sentences_in = autograd.Variable(sentences)
        targets = autograd.Variable(targets)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # step 3. Run forward pass
        labels = model(sentences_in, seq_lengths)
        error_count += (labels.data.max(1)[1] != targets.data).sum()
        total_count += BATCH_SIZE

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(labels, targets)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
    training_error_rates.append(error_count / total_count)
    test_error_rate = get_error_rate(model, training=False)
    test_error_rates.append(test_error_rate)
    acc = 1 - test_error_rate
    print('epoch: {}, time: {:.2f}s, cost so far: {}, accurary: {:.3f}'.format(
        start_epoch+epoch, (time.time() - since), total_loss.numpy(), acc))
    if acc > best_acc:
        save_checkpoint(model, acc, epoch)
        best_acc = acc

# save all_losses
save_to_pickle('checkpoint/all_losses.p', losses)
save_to_pickle('checkpoint/training_error_rates.p', training_error_rates)
save_to_pickle('checkpoint/test_error_rates.p', test_error_rates)
