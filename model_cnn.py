"""
Part of BME595 project
Program:
  CNN models for citation classification

Credit: adapted from https://github.com/xiayandi/Pytorch_text_classification/blob/master/cnn.py
by Yandi Xia
"""

import time  # debug
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import save_to_pickle, load_pickle


CONFIG = {
    "filter_sizes": [3],
    "num_filters": 50,   # could be 250
    # "vocab_size": None,
    # "emb_dim": 100,
    "hid_sizes": [50],  # could 250
    # "num_classes": 3,
    "dropout_switches": [True]
}


class CNN_NLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size,
                 vocab_size, num_classes, dropout=0.5, rnn_model="LSTM"):
        for arg in CONFIG:
            self.__setattr__(arg, CONFIG[arg])
        assert len(self.hid_sizes) == len(self.dropout_switches)
        super(CNN_NLP, self).__init__()

        self.batch_size = batch_size
        self.emb_dim = embedding_dim

        # make sure the padding word has embedding of 0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoders = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.num_filters,
                                       kernel_size=(filter_size, self.emb_dim)))
            self.encoders.append(self.__getattr__(enc_attr_name))

        self.hid_layers = []
        ins = len(self.filter_sizes) * self.num_filters
        for i, hid_size in enumerate(self.hid_sizes):
            hid_attr_name = "hid_layer_%d" % i
            self.__setattr__(hid_attr_name, nn.Linear(ins, hid_size))
            self.hid_layers.append(self.__getattr__(hid_attr_name))
            ins = hid_size
        self.logistic = nn.Linear(ins, num_classes)

    def forward(self, x):
        """
        :param x:
            input x is in size of [N, C, H, W]
            N: batch size
            C: number of channel, in text case, this is 1
            H: height, in text case, this is the length of the text
            W: width, in text case, this is the dimension of the embedding
        :return:
            a tensor [N, L], where L is the number of classes
        """
        n_idx = 0
        c_idx = 1
        h_idx = 2
        w_idx = 3
        # lookup table output size [N, H, W=emb_dim]
        x = self.embeddings(x)

        # expand x to [N, 1, H, W=emb_dim]
        x = x.unsqueeze(c_idx)

        enc_outs = []
        for encoder in self.encoders:
            enc_ = F.relu(encoder(x))
            k_h = enc_.size()[h_idx]
            k_w = 1
            enc_ = F.max_pool2d(enc_, kernel_size=(k_h, k_w))
            enc_ = enc_.squeeze(w_idx)
            enc_ = enc_.squeeze(h_idx)
            enc_outs.append(enc_)
        # each of enc_outs size [N, C]
        encoding = torch.cat(enc_outs, 1)
        hid_in = encoding
        for hid_layer, do_dropout in zip(self.hid_layers, self.dropout_switches):
            hid_out = F.relu(hid_layer(hid_in))
            if do_dropout:
                hid_out = F.dropout(hid_out, training=self.training)
            hid_in = hid_out
        pred_prob = F.log_softmax(self.logistic(hid_in))
        return pred_prob

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
