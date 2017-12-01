"""
Part of BME595 project
Program:
  Models for citation classification
"""
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from util import load_pickle, save_to_pickle


# does not support batch training
class LSTMCitationClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMCitationClassification, self).__init__()
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

class BatchLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size,
                 vocab_size, label_size, dropout=0.5):
        super(BatchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

        # make sure the padding word has embedding of 0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def load_glove_model(self, path_to_glove, word_to_idx,
                         saved_embedding='processed_data/glove_embedding.pkl',
                         regenerate=True):
        """
        Overwrite nn.Embedding.weight by pre-trained GloVe vectors.

        First load pre-trained GloVe model, i.e., a word-vector lookup table
        Then filter the words appeared in our dataset based on word_to_idx
        Then overwrite initial nn.Embedding.weight
        Credit: https://github.com/pytorch/text/issues/30
        """
        if regenerate:
            count = 0
            with open(path_to_glove, 'r') as f:
                for line in f.readlines():
                    # print(line)
                    row = line.split()
                    print(row)
                    word, vector = row[0], row[1:]
                    vector = torch.FloatTensor(list(map(float, vector)))
                    # only update the word that is in both word_to_idx and glove
                    # remain the same weight for the word that is not in glove model
                    if word in word_to_idx:
                        count += 1
                        # overwrite initial embedding.weight
                        self.embeddings.weight.data[word_to_idx[word]] = vector
                print('num of words in both word_to_idx and glove', count)
                save_to_pickle(saved_embedding, self.embeddings.weight.data)
        else:
            self.embeddings.weight.data.copy_(load_pickle(saved_embedding))

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentences, seq_lengths):
        """
        sentences: BxT, B is batch size, T is MAX_LEN
        seq_lengths: the length of sequences sorted by the number of non-padding words
        """
        # input is BxT, output of embedding is BxTxEmbed_dim
        embeds = self.embeddings(sentences)
        # pack, since batch_first is True, input is BxTxEmbed_dim
        pack = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=True)
        out, hidden = self.lstm(pack, self.hidden)
        # output is BxTxHidden_dim, T is seq_lengths[0], i.e., the max of seq_lengths
        unpack, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # only use the last time step output of LSTM for each sequence
        out_idxs = (autograd.Variable(torch.LongTensor(seq_lengths)) - 1).view(-1, 1).expand(unpack.size(0), unpack.size(2)).unsqueeze(1)
        last_outputs = unpack.gather(1, out_idxs).squeeze()
        # input to hidden2tag is BxHidden_dim
        label_space = self.hidden2tag(last_outputs)
        labels = F.log_softmax(label_space)
        return labels
