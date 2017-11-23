"""
Part of BME595 project
Program:
  Models for citation classification
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class LSTMCitationClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(self.__class__, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()        

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        out, hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        # label_space = self.hidden2tag(out.view(len(sentence), -1))
        # only use the last output of LSTM
        label_space = self.hidden2tag(out[-1])
        labels = F.log_softmax(label_space)
        return labels
