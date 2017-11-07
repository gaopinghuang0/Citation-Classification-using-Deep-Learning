"""
Part of Character-level RNN
To predict which language a name is from based on the spelling
Credit: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
import random
import time
from torch.autograd import Variable
from data import *


# random pick training sample
def randomTrainingPair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)

# a helper function to get category name and index
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i
