"""
Part of BME595 project
Program:
  Load data only once
"""
import torch
import torch.autograd as autograd
# import en    # not supported in python3
import string
import pickle

from nltk.corpus import stopwords
eng_stop = set(stopwords.words('english'))


def preprocess_data_large(max_len=60):
    polarity_to_idx = {'o': 0, 'p': 1, 'n': 2}
    polarities = []
    citing_sentences = []   # only the sentence with label 1
    word_to_idx = {}
    # database: http://cl.awaisathar.com/citation-sentiment-corpus/
    print('pre-processing data...')
    with open('./citation_sentiment_large/citation_sentiment_corpus.txt', 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or len(line.split('\t')) < 4:
                continue
            _, _, polarity, sent = line.strip().lower().split('\t')
            polarities.append(polarity_to_idx[polarity])
            # clean sentence by removing punctuations & stop words
            sent = my_tokenizer(sent)
            citing_sentences.append(sent[:max_len])  # truncate at length of max_len
    
    for sent in citing_sentences:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    
    # save as pickle for later use
    with open('processed_data/data_large.pkl', 'wb') as f:
        pickle.dump([citing_sentences, polarities, word_to_idx, polarity_to_idx], f)


def get_data_large():
    with open('processed_data/data_large.pkl', 'rb') as f:
        return pickle.load(f)

# Credit: https://stackoverflow.com/a/26802243
from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()
def unify_word(word):  # went -> go, apples -> apple, BIG -> big
    """unify verb tense and noun singular"""
    try: word = wordnet.lemmatize(word, 'v') # unify tense
    except: pass
    try: word = wordnet.lemmatize(word) # unify noun
    except: pass
    return word

# credit: https://stackoverflow.com/a/34294398
def remove_punctuation(s):
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator)

def remove_stopwords(s):
    return [w for w in s.split() if w not in eng_stop]

def my_tokenizer(s):
    s = remove_punctuation(s)
    s = s.lower()
    return [unify_word(w) for w in remove_stopwords(s)]


if __name__ == '__main__':
    preprocess_data_large()