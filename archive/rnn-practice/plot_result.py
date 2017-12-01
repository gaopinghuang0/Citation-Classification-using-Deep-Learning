"""
Part of Character-level RNN
To predict which language a name is from based on the spelling
Credit: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle

from data import *
from predict import evaluate
from util import randomTrainingPair, categoryFromOutput

def plot_loss():
    with open('all_losses.p', 'rb') as fp:
        all_losses = pickle.load(fp)

    plt.figure()
    plt.plot(all_losses)
    plt.show()

def plot_confusion_matrix():
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingPair()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # set up axes
    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

if __name__ == '__main__':
    plot_loss()
    # plot_confusion_matrix()
