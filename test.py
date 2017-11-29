
from data import get_data_small, get_data_large, get_combined_data
from collections import Counter


citing_sentences, polarities, word_to_idx, polarity_to_idx = get_combined_data()
print(len(citing_sentences))
print(Counter(polarities))

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_combined_data(balance_skew=False)
print(len(citing_sentences))

print(citing_sentences[0])
print(len(word_to_idx))

for sent in citing_sentences:
    print(sent)
