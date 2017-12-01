"""
Part of BME595 project
Program:
  Load data after preprocess
"""
# import en    # not supported in python3
import string
import pickle
import random
import re
import time
from collections import Counter
from constants import MAX_LEN
from util import save_to_pickle
# Credit: https://stackoverflow.com/a/26802243
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
ENG_STOP = set(stopwords.words('english'))


def _preprocess_data_small(max_len=60):
    # dataset: http://clair.si.umich.edu/corpora/citation_sentiment_umich.tar.gz
    purposes = []
    polarities = []
    citing_sentences = []   # only the sentence with label 1
    # context_sentences = []   # all the 4 sentences
    print('pre-processing data small...')
    with open('./raw_data/citation_sentiment_small/corpus.txt', 'r') as f:
        for line in f.readlines():
            _, _, _, s1, _, s2, _, s3, _, s4, _, purpose_label, polarity_label = line.split('\t')
            if int(polarity_label) == 0:
                continue
            purposes.append(int(purpose_label)-1)
            polarities.append(int(polarity_label)-1)
            citing_sentences.append(my_tokenizer(s2)[:max_len])
            # context_sentences.append([x.split() for x in [s1, s2, s3, s4]])
    return citing_sentences, polarities, None

def _preprocess_data_large(max_len=60):
    # dataset: http://cl.awaisathar.com/citation-sentiment-corpus/
    polarity_to_idx = {'o': 0, 'p': 1, 'n': 2}
    polarities = []
    citing_sentences = []   # only the sentence with label 1
    print('pre-processing data large...')
    with open('./raw_data/citation_sentiment_large/corpus.txt', 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or len(line.split('\t')) < 4:
                continue
            _, _, polarity, sent = line.strip().lower().split('\t')
            polarities.append(polarity_to_idx[polarity])
            # clean sentence by replacing punctuations & stop words
            sent = my_tokenizer(sent)
            citing_sentences.append(sent[:max_len])  # truncate at length of max_len
    return citing_sentences, polarities, polarity_to_idx

def preprocess_data_small(max_len=60):
    citing_sentences, polarities, polarity_to_idx = _preprocess_data_small(max_len)
    word_to_idx = compute_word_to_idx(citing_sentences)
    # save as pickle for later use
    save_to_pickle('processed_data/data_small.pkl',
                   [citing_sentences, polarities, word_to_idx, polarity_to_idx])

def preprocess_data_large(max_len=60):
    citing_sentences, polarities, polarity_to_idx = _preprocess_data_large(max_len)
    word_to_idx = compute_word_to_idx(citing_sentences)
    # save as pickle for later use
    save_to_pickle('processed_data/data_large.pkl',
                   [citing_sentences, polarities, word_to_idx, polarity_to_idx])

def preprocess_data_combined(max_len=60):
    small_sentences, small_polarities, _ = _preprocess_data_small(max_len)
    large_sentences, large_polarities, polarity_to_idx = _preprocess_data_large(max_len)
    combined_sentences = small_sentences + large_sentences
    combined_polarities = small_polarities + large_polarities
    # shuffle all data
    data = list(zip(combined_sentences, combined_polarities))
    random.shuffle(data)
    word_to_idx = compute_word_to_idx(combined_sentences)
    combined_sentences, combined_polarities = zip(*data)
    # save as pickle for later use
    save_to_pickle('processed_data/data_combined.pkl',
                   [combined_sentences, combined_polarities, word_to_idx, polarity_to_idx])


def compute_word_to_idx(sentences):
    ctr = Counter()
    for sent in sentences:
        ctr += Counter(sent)
    words_sorted = sorted(ctr, key=ctr.get, reverse=True)
    word_to_idx = {word: i for i, word in enumerate(words_sorted, 1)}
    word_to_idx['<PAD>'] = 0
    return word_to_idx


def get_combined_data(training=True, portion=0.85, balance_skew=True):
    return _get_data('processed_data/data_combined.pkl', training, portion, balance_skew)

def get_data_large(training=True, portion=0.85, balance_skew=True):
    return _get_data('processed_data/data_large.pkl', training, portion, balance_skew)

def get_data_small(training=True, portion=0.85, balance_skew=True):
    return _get_data('processed_data/data_small.pkl', training, portion, balance_skew)

def _get_data(filename, training=True, portion=0.85, balance_skew=True):
    with open(filename, 'rb') as f:
        citing_sentences, polarities, word_to_idx, polarity_to_idx = pickle.load(f)
        end = int(len(citing_sentences) * portion)
        sentences = citing_sentences[:end] if training else citing_sentences[end:]
        polarities = polarities[:end] if training else polarities[end:]
        if balance_skew:
            # keep the number of neutral similar to the number of positive+negative
            ctr = Counter(polarities)
            size = ctr[1] + ctr[2]
            balanced_sentences = []
            balanced_polarities = []
            for sent, polarity in zip(sentences, polarities):
                if polarity == 0:
                    size -= 1
                    if size < 0:
                        continue
                balanced_sentences.append(sent)
                balanced_polarities.append(polarity)
            return balanced_sentences, balanced_polarities, word_to_idx, polarity_to_idx
        return sentences, polarities, word_to_idx, polarity_to_idx



wordnet = WordNetLemmatizer()
def unify_word(word):  # went -> go, apples -> apple, BIG -> big
    """unify verb tense and noun singular"""
    try:
        word = wordnet.lemmatize(word, 'v') # unify tense
    except:
        pass
    try:
        word = wordnet.lemmatize(word) # unify noun
    except:
        pass
    return word

def remove_xml_tags(seq):
    """
    Remove the XML-like tags, such as <REF>, <TREF>, <marker>, ...
    Note that tags should not have whitespace right after '<' and right before '>'
    """
    ptn = r'<\S[^>]*\S>'  # FIXME: it doesn't work for case like '<a>'
    return re.sub(ptn, '', seq)

# credit: https://stackoverflow.com/a/34294398
def remove_punctuation(seq):
    translator = str.maketrans('', '', string.punctuation)
    return seq.translate(translator)

# Credit: https://stackoverflow.com/a/12437721
def replace_punctuation_with_space(seq):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', seq)

def remove_stopwords(seq):
    return [w for w in seq.split() if w not in ENG_STOP]

def my_tokenizer(seq):
    seq = remove_xml_tags(seq)
    seq = replace_punctuation_with_space(seq)
    seq = seq.lower()
    return [unify_word(w) for w in remove_stopwords(seq)]


if __name__ == '__main__':
    # preprocess_data_large(MAX_LEN)
    # preprocess_data_small(MAX_LEN)
    preprocess_data_combined(MAX_LEN)
