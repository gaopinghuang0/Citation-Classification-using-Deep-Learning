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

from model import BatchLSTM
from data import get_data_large
from util import prepare_sequence, label_to_text, get_batch_data

import time
import random

torch.manual_seed(1)

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 8
BATCH_SIZE = 20
EPOCHS = 10
print('total epochs: ', EPOCHS)


losses = []
loss_function = nn.NLLLoss()
model = BatchLSTM(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, len(word_to_idx), len(polarity_to_idx))
optimizer = optim.SGD(model.parameters(), lr=0.1)


# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# inputs = prepare_sequence(citing_sentences[0], word_to_idx)
# print(inputs)
# labels = model(inputs)
# print(labels)
# label_to_text(labels, polarity_to_idx)
since = time.time()
training_data = list(zip(citing_sentences, polarities))
for epoch in range(EPOCHS):
    total_loss = torch.Tensor([0])
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

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(labels, targets)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    print('epoch: {}, time: {:.2f}s, cost so far: {}'.format(epoch, (time.time() - since), total_loss))
    losses.append(total_loss)

# save model
torch.save(model, 'lstm-citation-classification.ckpt')
# save all_losses
import pickle
with open('all_losses.p', 'wb') as fp:
    pickle.dump(losses, fp)
