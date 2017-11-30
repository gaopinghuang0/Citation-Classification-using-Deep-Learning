
import time

import torch
from data import get_data_large, remove_xml_tags, my_tokenizer
from util import get_batch_data

torch.manual_seed(1)

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()
training_data = list(zip(citing_sentences, polarities))


# for sentences, targets, seq_lengths in get_batch_data(training_data, 20, word_to_idx, shuffle=True):
#     print(sentences[0], seq_lengths)
#     time.sleep(50)

x = "So far, most of the statistical machine translation systems are based on the single-word alignment models as described in (Brown et al. , 1993) as well as the Hidden Markov alignment model (Vogel et al. , 1996)."
print(x)
print(remove_xml_tags(x))
print(my_tokenizer(x))