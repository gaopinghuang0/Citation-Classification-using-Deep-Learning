"""
Part of BME595 project
Program:
  Classify citation  models for citation classification
"""

import torch
import time
from collections import Counter
from data import citing_sentences, polarities, word_to_idx
from util import prepare_sequence

model = torch.load('lstm-citation-classification.ckpt')

# Just return an table
def evaluate(sentence):
    inputs = prepare_sequence(sentence, word_to_idx)
    labels = model(inputs)
    return labels.data.max(1)[1][0]

def correct_rate():
    test_data = list(zip(citing_sentences, polarities))[100:200]
    count = 0
    ctr_p = Counter()
    ctr_t = Counter()
    for sentence, target in test_data:
        predict = evaluate(sentence)
        ctr_p[predict] += 1
        ctr_t[target] += 1
        if predict == target:
            count += 1
    print(ctr_p, ctr_t)
    print('correct rate: ', count / len(test_data))


# # See what the scores are after training

# # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
# #  for word i. The predicted tag is the maximum scoring tag.
# # Here, we can see the predicted sequence below is 0 1 2 0 1
# # since 0 is index of the maximum value of row 1,
# # 1 is the index of maximum value of row 2, etc.
# # Which is DET NOUN VERB DET NOUN, the correct sequence!
# print(labels)
# print(citing_sentences[0])
# label_to_text(labels, polarity_to_idx)

# # print(losses)
if __name__ == '__main__':
    correct_rate()