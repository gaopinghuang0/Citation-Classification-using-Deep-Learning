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

from model import LSTMCitationClassification
from data import *
from util import prepare_sequence, label_to_text

import time
import random

torch.manual_seed(1)


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

losses = []
loss_function = nn.NLLLoss()
model = LSTMCitationClassification(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(polarity_to_idx))
optimizer = optim.SGD(model.parameters(), lr=0.1)



# # See what the scores are before training
# # Note that element i,j of the output is the score for tag j for word i.
# inputs = prepare_sequence(citing_sentences[0], word_to_idx)
# print(inputs)
# labels = model(inputs)
# print(labels)
# label_to_text(labels, polarity_to_idx)
since = time.time()
training_data = list(zip(citing_sentences, polarities))[:100]
for epoch in range(10):
    total_loss = torch.Tensor([0])
    i = 0
    random.shuffle(training_data)
    for sentence, target in training_data:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        sentence_in = prepare_sequence(sentence, word_to_idx)
        target = autograd.Variable(torch.LongTensor([target]))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # step 3. Run forward pass
        labels = model(sentence_in)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(labels, target)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        if i % 100 == 0:
            print(total_loss, '%.2fs'%(time.time() - since))
        i += 1
    print('epoch ', epoch, 'cost so far:', total_loss)
    losses.append(total_loss)

# save model
torch.save(model, 'lstm-citation-classification.ckpt')
# save all_losses
import pickle
with open('all_losses.p', 'wb') as fp:
    pickle.dump(losses, fp)
