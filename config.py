"""
Part of BME595 project
Program:
  Some constants
"""

MAX_LEN = 60   # remember to rerun preprocess if changed
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
BATCH_SIZE = 30
EPOCHS = 30

RUN_MODE = 'CNN'    # can be CNN, RNN, LSTM, GRU
DATASET_MODE = 'purpose'  # can be polarity or purpose
MODEL_FILENAME = 'checkpoint/{}_{}_classification.ckpt'.format(RUN_MODE.lower(), DATASET_MODE)

MERGE_POS_NEG = False  # if true, treat positive and negative polarity as one class
