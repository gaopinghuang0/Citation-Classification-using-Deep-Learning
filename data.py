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
import config as cfg
from util import save_to_pickle
# Credit: https://stackoverflow.com/a/26802243
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
ENG_STOP = set(stopwords.words('english'))

random.seed(123)

def _preprocess_dataset_small(max_len=60, augmentation=False, deduplicate=True):
    # dataset: http://clair.si.umich.edu/corpora/citation_sentiment_umich.tar.gz
    purposes = []
    polarities = []
    citing_sentences = []   # only the sentence with label 1
    # context_sentences = []   # all the 4 sentences
    augment_sentences = []
    seen = {}
    print('pre-processing data small...')
    with open('./raw_data/citation_sentiment_small/corpus.txt', 'r') as f:
        for line in f.readlines():
            _, _, _, s1, _, s2, _, s3, _, s4, _, purpose_label, polarity_label = line.split('\t')
            if int(polarity_label) == 0:   # invalid label
                continue
            sent = my_tokenizer(s2)[:max_len]
            key = ''.join(sent)  
            if deduplicate and key in seen:
                continue
            seen[key] = True
            citing_sentences.append(sent)
            purposes.append(int(purpose_label)-1)
            polarities.append(int(polarity_label)-1)

            if augmentation:   # use non-unified word as augmentation
                augment_sentences.append(my_tokenizer(s2, should_unify=False)[:max_len])
            # context_sentences.append([x.split() for x in [s1, s2, s3, s4]])
    print('deduplicate:', deduplicate, '; unique sentences:', len(seen))
    assert(len(citing_sentences) == len(polarities) == len(purposes))
    return citing_sentences, polarities, purposes, augment_sentences

def _preprocess_dataset_large(max_len=60, deduplicate=True):
    # dataset: http://cl.awaisathar.com/citation-sentiment-corpus/
    polarity_to_idx = {'o': 0, 'p': 1, 'n': 2}
    polarities = []
    citing_sentences = []   # only the sentence with label 1
    seen = {}
    print('pre-processing data large...')
    with open('./raw_data/citation_sentiment_large/corpus.txt', 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or len(line.split('\t')) < 4:
                continue
            _, _, polarity, sent = line.strip().lower().split('\t')
            # clean sentence by replacing punctuations & stop words
            sent = my_tokenizer(sent)[:max_len]  # truncate at length of max_len
            key = ''.join(sent)
            if deduplicate and key in seen:
                continue
            seen[key] = True
            citing_sentences.append(sent)  
            polarities.append(polarity_to_idx[polarity])
    print('deduplicate:', deduplicate, '; unique sentences:', len(seen))
    assert(len(citing_sentences) == len(polarities))
    return citing_sentences, polarities, polarity_to_idx

def preprocess_purpose_data(max_len=60, portion=0.85, augmentation=False, deduplicate=True):
    sentences, _, purposes, augment_sentences = _preprocess_dataset_small(max_len, augmentation, deduplicate=deduplicate)
    # shuffle all data
    if augmentation:
        data = list(zip(sentences, purposes, augment_sentences))
        word_to_idx = compute_word_to_idx(sentences+augment_sentences)
    else:
        data = list(zip(sentences, purposes))
        word_to_idx = compute_word_to_idx(sentences)

    random.shuffle(data)
    random.shuffle(data)
    end = int(len(data) * portion)
    train_data, test_data = data[:end], data[end:]

    if augmentation:
        # only augment train_data, concat both sentences
        train_sentences, train_purposes, train_augment_sentences = zip(*train_data)
        train_sentences += train_augment_sentences
        train_purposes += train_purposes
        train_data = list(zip(train_sentences, train_purposes))
        # rebind test_data
        test_sentences, test_purposes, test_augment_sentences = zip(*test_data)
        test_data = list(zip(test_sentences, test_purposes))

    # save as pickle for later use
    save_to_pickle('processed_data/purpose.train.pkl', [train_data, word_to_idx])
    save_to_pickle('processed_data/purpose.test.pkl', [test_data, word_to_idx])

def preprocess_polarity_data(max_len=60, portion=0.85, deduplicate=True):
    small_sentences, small_polarities, _, _ = _preprocess_dataset_small(max_len, deduplicate=deduplicate)
    large_sentences, large_polarities, polarity_to_idx = _preprocess_dataset_large(max_len, deduplicate=deduplicate)
    combined_sentences = small_sentences + large_sentences
    combined_polarities = small_polarities + large_polarities
    data = []
    if deduplicate:
        seen = {}
        for sent, polarity in zip(combined_sentences, combined_polarities):
            key = ''.join(sent)
            if key not in seen:
                seen[key] = True
                data.append((sent, polarity))
        print('unique sentences:', len(seen), 'duplicate:', len(combined_sentences)-len(seen))
    else:
        data = list(zip(combined_sentences, combined_polarities))
    
    # shuffle all data
    random.shuffle(data)
    word_to_idx = compute_word_to_idx(combined_sentences)
    end = int(len(data) * portion)
    train_data, test_data = data[:end], data[end:]

    # save as pickle for later use
    save_to_pickle('processed_data/polarity.train.pkl', [train_data, word_to_idx, polarity_to_idx])
    save_to_pickle('processed_data/polarity.test.pkl', [test_data, word_to_idx, polarity_to_idx])


def compute_word_to_idx(sentences):
    ctr = Counter()
    for sent in sentences:
        ctr += Counter(sent)
    words_sorted = sorted(ctr, key=ctr.get, reverse=True)
    word_to_idx = {word: i for i, word in enumerate(words_sorted, 1)}
    word_to_idx['<PAD>'] = 0
    return word_to_idx

def data_loader(training=True, balance_skew=True, dataset_mode=cfg.DATASET_MODE):
    if dataset_mode == 'polarity':
        data, word_to_idx, label_to_idx = polarity_data_loader(training, balance_skew)

        if cfg.MERGE_POS_NEG:
            label_to_idx = {'neutral': 0, 'subjective': 1}
    else:
        data, word_to_idx, label_to_idx = purpose_data_loader(training, balance_skew)
    return data, word_to_idx, label_to_idx

def purpose_data_loader(training=True, balance_skew=True):
    filename = 'processed_data/purpose.{}.pkl'.format('train' if training else 'test')
    with open(filename, 'rb') as fp:
        data, word_to_idx = pickle.load(fp)
        purpose_to_idx = {'Criticizing': 0, 'Comparison': 1, 'Use': 2,
                          'Substantiating': 3, 'Basis': 4, 'Neutral': 5}
        return data, word_to_idx, purpose_to_idx

def polarity_data_loader(training=True, balance_skew=True):
    filename = 'processed_data/polarity.{}.pkl'.format('train' if training else 'test')
    with open(filename, 'rb') as fp:
        data, word_to_idx, polarity_to_idx = pickle.load(fp)
        if balance_skew:
            # keep the number of neutral similar to the number of positive+negative
            ctr = 0
            balanced_data = []
            for sent, polarity in data:
                if polarity == 0:  # neutral
                    if ctr < 0:
                        continue
                    ctr -= 1
                else:
                    ctr += 1
                balanced_data.append((sent, polarity))
            return balanced_data, word_to_idx, polarity_to_idx
        return data, word_to_idx, polarity_to_idx


wordnet = WordNetLemmatizer()
def unify_word(word):  # went -> go, apples -> apple, BIG -> big
    """unify verb tense and noun singular"""
    # return word  # compare w/ or w/o unify
    try:
        word = wordnet.lemmatize(word, 'v') # unify tense
    except e:
        print(e)
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
    return (w for w in seq.split() if w not in ENG_STOP)

def my_tokenizer(seq, should_unify=True):
    seq = remove_xml_tags(seq)
    seq = replace_punctuation_with_space(seq)
    seq = seq.lower()
    if should_unify:
        return [unify_word(w) for w in remove_stopwords(seq)]
    return list(remove_stopwords(seq))


if __name__ == '__main__':
    # preprocess_purpose_data(cfg.MAX_LEN, augmentation=True, deduplicate=False)
    preprocess_polarity_data(cfg.MAX_LEN, deduplicate=False)
