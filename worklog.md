# Things we have done

## Increase accurary
* [DONE] clean data: 
	* remove tags like <REF> and <TREF>, or even remove the content between these tags
	* replace punctuations & remove stop words
	* manually split some mistakenly concatenated words
* [DONE] draw histogram for sentence length
* [DONE] Balance skewed data: keep similar size of neural and positive+negative
* [Not working] Order word_to_idx by frequency
* [DONE] add dropout
* [DONE] Try learning rate
* [DONE] combine dataset_small and dataset_large, and then shuffle
* [Not working as the model is overfitting now] Bi-LSTM, two layer LSTM, Bi-LSTM-CRF
* [DONE] Adjust EMBEDDING_DIM, HIDDEN_DIM
* [DONE large, todo small] Add space to invalid words in dataset
* [DONE] Use existing word embeddings (word2vec, GloVe) instead of torch.nn.Embeddings
* [DONE, not working] first compare neutral vs. subjective, then compare positive vs. negative
* [DONE] try diff loss function
* [DONE] try diff backprop optimization
* [DONE, Not working] duplicate the data of positive and negative to make size balance
* Normalize data
* Add char-level RNN layer

## Speed up training
* [DONE] Increase batch_size
* [DONE] Limit the sentence length to max_len:
	* [Recurrent Models with sequences of mixed length](https://github.com/fchollet/keras/issues/40)
	* zero_padding to the ~~left~~ right of the shorter sentence, and use `pack_padded_sequence()`, see [handle variable length inputs sentences](https://discuss.pytorch.org/t/how-to-handle-variable-length-inputs-sentences/5407)
	> I am using my own pre-trained word embeddings and i apply zero_padding (to the right) on all sentences. The problem is that with my current code, the LSTM processes all timesteps, even the zero padded. How can i modify my code to handle variable length inputs?

	Our understanding is that when we use 
	`nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)`, the RNN model will stop `forward` pass based on the `seq_length` of each sequence. Therefore, it will not involve the padding zeros.
	* truncate the right of longer sentence
	* No need to use mask, [tweet sentiment analyzer](http://deeplearning.net/tutorial/code/lstm.py). Based on [seq2seq loss using mask](https://discuss.pytorch.org/t/how-can-i-compute-seq2seq-loss-using-mask/861):
	> In seq2seq, padding is used to handle the variable-length sequence problems. Additionally, mask is multiplied by the calculated loss (vector not scalar) so that the padding does not affect the loss.


## Useful Blogs
1. [Sentiment analysis using RNNs(LSTM)](https://towardsdatascience.com/sentiment-analysis-using-rnns-lstm-60871fa6aeba)
	> One thing in my experiments I could not explain is when I encode the words to integers if I randomly assign unique integers to words the best accuracy I get is 50–55% (basically the model is not doing much better than random guessing). However if the words are encoded such that highest frequency words get the lowest number then the model accuracy is 80% in 3–5 epochs. My guess is this is necessary to train the embedding layer but cannot find an explanation on why anywhere.
2. [handle variable length inputs sentences](https://discuss.pytorch.org/t/how-to-handle-variable-length-inputs-sentences/5407)
3. [Multiple PackedSequence input ordering](https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106)
	> pytorch to get the last time step for each sequence
	```python
    idx = (seq_sizes - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1)
    decoded = output.gather(1, idx).squeeze()
	```
4. [How to Handle Very Long Sequences with Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)
	> LSTMs work very well if your problem has one output for every input, like time series forecasting or text translation. But LSTMs can be challenging to use when you have very long input sequences and only one or a handful of outputs. 
	> This is often called sequence labeling, or sequence classification.
