"""
Part of BME595 project
Program:
  Some util functions
"""
import torch
import torch.autograd as autograd

# print(word_to_idx)

def prepare_sequence(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return autograd.Variable(torch.LongTensor(idxs))

def label_to_text(scores, label_to_idx):
    ix2tag = {i:v for v, i in label_to_idx.items()}
    tensor = scores.data
    print([ix2tag[i] for i in tensor.max(1)[1]])



if __name__ == '__main__':
    pass