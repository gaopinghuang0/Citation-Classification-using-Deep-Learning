
from collections import Counter

from data import data_loader
from util import load_pickle

data, word_to_idx, label_to_idx = data_loader()
print(len(data))

data, word_to_idx, label_to_idx = data_loader(balance_skew=False)
print(len(data))

print(data[:2])
print(len(word_to_idx))


# print word by length in reverse order
# print('\n'.join(sorted(word_to_idx.keys(), key=len, reverse=True)[:100]))

# path_to_glove = 'GloVe-1.2/vectors.txt'
# with open(path_to_glove, 'r') as f:
#     glove_words = set((line.split()[0] for line in f.readlines()))
#     our_words = set(word_to_idx.keys())
#     print(len(glove_words.intersection(our_words)))
#     # get word that is in word_to_idx but not in Glove
#     print(len(our_words.difference(glove_words)))
#     print(our_words.difference(glove_words))