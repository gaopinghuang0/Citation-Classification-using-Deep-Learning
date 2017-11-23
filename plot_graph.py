"""
Part of BME595 project
Program:
  Plot graphs
"""
import torch
import matplotlib.pyplot as plt

from data import get_data_large


citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()

def histogram_sentence_length(sentences):
    lens = list(map(len, sentences))
    plt.hist(lens)
    plt.title("Sentence Length Histogram")
    plt.xlabel('Len')
    plt.ylabel('count')
    plt.show()


if __name__ == '__main__':
    histogram_sentence_length(citing_sentences)
