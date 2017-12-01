"""
Part of BME595 project
Program:
  Load data only once
"""
import torch
import torch.autograd as autograd

# database: http://clair.si.umich.edu/corpora/citation_sentiment_umich.tar.gz
purposes = []
polarities = []
citing_sentences = []   # only the sentence with label 1
context_sentences = []   # all the 4 sentences
with open('./citation_sentiment_small/annotated_sentences.txt', 'r') as f:
    for line in f.readlines():
        _, _, _, s1, _, s2, _, s3, _, s4, _, purpose_label, polarity_label = line.split('\t')
        if int(polarity_label) == 0:
            continue
        purposes.append(int(purpose_label)-1)
        polarities.append(int(polarity_label)-1)
        citing_sentences.append(s2.split())
        context_sentences.append([x.split() for x in [s1, s2, s3, s4]])


word_to_idx = {}
for sent in citing_sentences:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

polarity_to_idx = {'neutral': 0, 'positive': 1, 'negative': 2}




if __name__ == '__main__':
    pass