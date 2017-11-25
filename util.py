"""
Part of BME595 project
Program:
  Some util functions
"""
import torch
import torch.autograd as autograd
import random
from constants import *
# print(word_to_idx)

def prepare_sequence(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return autograd.Variable(torch.LongTensor(idxs))

def label_to_text(scores, label_to_idx):
    ix2tag = {i:v for v, i in label_to_idx.items()}
    tensor = scores.data
    print([ix2tag[i] for i in tensor.max(1)[1]])


def get_batch_data(data, batch_size, word_to_idx, padding_len=MAX_LEN, shuffle=True):
    if shuffle:
        random.shuffle(data)
    size = len(data)
    n = size // batch_size
    for i in range(n):
        start = i * batch_size
        sentences = []
        targets = []
        batch_input = data[start:start+batch_size]
        # sort by length
        batch_input.sort(key=lambda x: -len(x[0]))
        seq_lengths = [len(x) for x,_ in batch_input]
        for sentence, target in batch_input:
            idxs = [word_to_idx[w] for w in sentence]
            sentences.append(zero_padding(idxs, 0, padding_len))
            targets.append(target)
        yield torch.LongTensor(sentences), torch.LongTensor(targets), seq_lengths

def zero_padding(s, padding='<PAD>', max_len=60):
    # insert padding string to the right of s
    if len(s) < max_len:
        return s + [padding]*(max_len - len(s))
    return s


if __name__ == '__main__':
    pass