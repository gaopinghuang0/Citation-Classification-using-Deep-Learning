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
MODEL_FILENAME = 'checkpoint/{}_text_classification.ckpt'.format(RUN_MODE.lower())

MERGE_POS_NEG = True
