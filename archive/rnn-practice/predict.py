"""
Part of Character-level RNN
To predict which language a name is from based on the spelling
Credit: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import sys
from data import *
from model import *

rnn = torch.load('char-rnn-classification.ckpt')

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_i = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_i]))
        predictions.append([value, all_categories[category_i]])

    return predictions

if __name__ == '__main__':
    predict(sys.argv[1])