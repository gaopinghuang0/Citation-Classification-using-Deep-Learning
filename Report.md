

## must TODOs
1. Try learning rate
2. Order word_to_idx by frequency
3. clean data: remove punctuations & stop words
4. Balance skewed data: keep similar size of neural and positive+negative
* draw histogram for sentence length

## Speed up training
* Increase batch_size, may not work for mixed length of sentenses, [Recurrent Models with sequences of mixed length](https://github.com/fchollet/keras/issues/40)
* Limit the sentence length
 

## Possible todos:
1. 2-layer LSTM
2. use a fixed length, such as masking, [tweet sentiment analyzer](http://deeplearning.net/tutorial/code/lstm.py)
3. [How to Handle Very Long Sequences with Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
	> LSTMs work very well if your problem has one output for every input, like time series forecasting or text translation. But LSTMs can be challenging to use when you have very long input sequences and only one or a handful of outputs. 
	> This is often called sequence labeling, or sequence classification.
4. Use existing word embeddings (word2vec, GloVe) instead of torch.nn.Embeddings

## Useful Blogs
1. [Sentiment analysis using RNNs(LSTM)](https://towardsdatascience.com/sentiment-analysis-using-rnns-lstm-60871fa6aeba)
	> One thing in my experiments I could not explain is when I encode the words to integers if I randomly assign unique integers to words the best accuracy I get is 50–55% (basically the model is not doing much better than random guessing). However if the words are encoded such that highest frequency words get the lowest number then the model accuracy is 80% in 3–5 epochs. My guess is this is necessary to train the embedding layer but cannot find an explanation on why anywhere.


