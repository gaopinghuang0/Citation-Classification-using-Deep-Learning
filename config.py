"""
Part of BME595 project
Program:
  Some constants and configs
"""

MAX_LEN = 60   # remember to rerun preprocess if changed
HIDDEN_DIM = 100
BATCH_SIZE = 30
EPOCHS = 30

EMBEDDING_DIM = 100
USE_GLOVE = True
# GLOVE_FILE = 'GloVe-1.2/glove.6B.100d.txt'
GLOVE_FILE = 'GloVe-1.2/domain_specific.txt'

RUN_MODE = 'CNN'    # can be CNN, RNN, LSTM, GRU
DATASET_MODE = 'purpose'  # can be polarity or purpose
MODEL_FILENAME = 'checkpoint/{}_{}_classification.ckpt'.format(RUN_MODE.lower(), DATASET_MODE)

MERGE_POS_NEG = False  # if true, treat positive and negative polarity as one class
