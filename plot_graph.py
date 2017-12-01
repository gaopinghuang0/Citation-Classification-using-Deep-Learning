"""
Part of BME595 project
Program:
  Plot graphs
"""
import torch
import matplotlib.pyplot as plt
import pickle

import time
from data import get_data_large
from util import load_pickle, get_local_time_string


def draw_sentence_length_histogram(sentences):
    lens = list(map(len, sentences))
    plt.hist(lens)
    plt.title("Sentence Length Histogram")
    plt.xlabel('Len')
    plt.ylabel('count')
    plt.show()


def plot_loss():
    all_losses = load_pickle('checkpoint/all_losses.p')
    plt.figure()
    plt.plot([x.numpy() for x in all_losses])
    plt.show()

def plot_error_rates():
    training_error_rates = load_pickle('checkpoint/training_error_rates.p')
    test_error_rates = load_pickle('checkpoint/test_error_rates.p')
    my_lineplot(range(1,len(test_error_rates)+1), [training_error_rates, test_error_rates],
        filename="figures/error_rate_{}.png".format(get_local_time_string()),
        legend=['training', 'test'],
        xlabel="epoch", ylabel="error_rate",
        title="training vs. test error rate"
    )

def my_lineplot(x_axis, y_axis_list, filename=None, legend=None, xlabel=None, ylabel=None, title=None):
    """
    Draw multiple line plots (specified by y_axis_list) over the same x-axis.

    :type x_axis: [int]
    :type y_axis_list: [[float]]
    :type legend: [str]
    """
    res = []
    for y_axis in y_axis_list:
        print(x_axis, y_axis)
        line_plt, = plt.plot(x_axis, y_axis)
        res.append(line_plt)
    if title:
        plt.title(title)
    if legend:
        plt.legend(res, legend)
    if xlabel and ylabel:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if filename:
        plt.savefig(filename)
    plt.show()  # must show, otherwise, the current plt will affect later plt


if __name__ == '__main__':
    citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()
    # draw_sentence_length_histogram(citing_sentences)
    # plot_loss()
    plot_error_rates()
