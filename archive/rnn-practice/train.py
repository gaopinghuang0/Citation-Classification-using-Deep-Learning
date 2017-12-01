"""
Part of Character-level RNN
To predict which language a name is from based on the spelling
Credit: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import torch

from data import *
from util import *
from model import *

n_hidden = 128
n_epochs = 100000   # about 5 mins in total
print_every = 5000
plot_every = 1000
learning_rate = 0.005


rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()
    return output, loss.data[0]

# keep track of loss for plotting
current_loss = 0
all_losses = []

start = time.time()
for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

# save model
torch.save(rnn, 'char-rnn-classification.ckpt')
# save all_losses
import pickle
with open('all_losses.p', 'wb') as fp:
    pickle.dump(all_losses, fp)

# to read it back
# with open('all_losses.p', 'rb') as fp:
#     all_losses = pickle.load(fp)
