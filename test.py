
from data import get_data_large

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()
print(len(citing_sentences))

citing_sentences, polarities, word_to_idx, polarity_to_idx = get_data_large()
print(len(citing_sentences))

print(citing_sentences[0])
print(len(word_to_idx))


from collections import Counter
ctr = Counter(map(len, citing_sentences))
for c in ctr:
    if ctr[c] <= 2:
        print(c)