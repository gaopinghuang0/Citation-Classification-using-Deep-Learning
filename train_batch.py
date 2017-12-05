"""
Part of BME595 project
Program:
  Train models for citation classification
"""
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


from model import BatchRNN
from model_cnn import CNN_NLP
from data import get_combined_data
from util import get_batch_data, save_to_pickle, load_checkpoint, save_checkpoint
from predict import get_error_rate
from config import *

torch.manual_seed(1)

def get_model(word_to_idx, polarity_to_idx, resume=False, use_glove=True):
    """Resume a saved model or build a new model"""

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if resume:
        # load checkpoint
        checkpoint = load_checkpoint()
        model = checkpoint['model']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model {}...'.format(RUN_MODE))
        if RUN_MODE in ["RNN", "LSTM", "GRU"]:
            model = BatchRNN(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE,
                             len(word_to_idx), len(polarity_to_idx), rnn_model=RUN_MODE)
        else:
            model = CNN_NLP(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE,
                            len(word_to_idx), len(polarity_to_idx))
        if use_glove:
            # model.load_glove_model('GloVe-1.2/vectors.txt', word_to_idx)
            model.load_glove_model('GloVe-1.2/glove.6B.100d.txt', word_to_idx, regenerate=False)
    return model, best_acc, start_epoch

def train(model, loss_function, optimizer, training_data, word_to_idx):
    """Train one epoch of mini-batch"""
    if RUN_MODE == 'CNN':
        model.train()

    train_loss = torch.Tensor([0])
    error_count = 0
    total_count = 0
    for sentences, targets, seq_lengths in get_batch_data(training_data, BATCH_SIZE,
                                                          word_to_idx, shuffle=True):
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        sentences_in = autograd.Variable(sentences)
        targets = autograd.Variable(targets)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # step 3. Run forward pass
        if RUN_MODE == 'CNN':
            labels = model(sentences_in)
        else:
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()
            labels = model(sentences_in, seq_lengths)

        error_count += (labels.data.max(1)[1] != targets.data).sum()
        total_count += BATCH_SIZE

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(labels, targets)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()
        train_loss += loss.data
    return error_count / total_count, train_loss


def train_epochs(resume=False, use_glove=True):
    """Train multiple opochs"""

    print('total epochs: ', EPOCHS, '; use_glove: ', use_glove)

    citing_sentences, polarities, word_to_idx, polarity_to_idx = get_combined_data()

    model, best_acc, start_epoch = get_model(word_to_idx, polarity_to_idx,
                                             resume, use_glove)

    losses = []
    loss_function = nn.NLLLoss()
    if RUN_MODE == 'CNN':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
    # optimizers below are not working
    # optimizer = optim.Adagrad(model.parameters(), lr=0.001)

    since = time.time()
    training_data = list(zip(citing_sentences, polarities))
    training_error_rates = []
    test_error_rates = []
    for epoch in range(1, EPOCHS+1):
        train_error, train_loss = train(model, loss_function,
                                        optimizer, training_data, word_to_idx)
        losses.append(train_loss)
        training_error_rates.append(train_error)
        test_error_rate = get_error_rate(model, training=False)
        test_error_rates.append(test_error_rate)
        acc = 1 - test_error_rate
        print('epoch: {}, time: {:.2f}s, cost so far: {}, accurary: {:.3f}'.format(
            start_epoch+epoch, (time.time() - since), train_loss.numpy(), acc))
        if acc > best_acc:
            save_checkpoint(model, acc, epoch)
            best_acc = acc

    # save all_losses
    save_to_pickle('checkpoint/all_losses.p', losses)
    save_to_pickle('checkpoint/training_error_rates.p', training_error_rates)
    save_to_pickle('checkpoint/test_error_rates.p', test_error_rates)


if __name__ == '__main__':
    train_epochs(resume=False, use_glove=True)
