## RNN example using PyTorch
This reproduces the RNN example in the PyTorch tutorial by Sean Robertson. 
- Tutorial: [Classifying Names with a Character-Level RNN](http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- Code: [practical-pytorch/char-rnn-classification](https://github.com/spro/practical-pytorch)

I restructured the code and added two files `util.py` and `plot_result.py`.

### Running on user input
```bash
$ python predict.py Hinton
(-0.47) Scottish
(-1.52) English
(-3.57) Irish

$ python predict.py Huang
(-0.40) Chinese
(-1.65) Vietnamese
(-2.61) Korean
```

### Plots
1. Training loss on total epochs = 100000

![training_loss](figures/train_loss.png)

2. Confusion matrix on 10000 random examples

![confusion_matrix](figures/confusion_matrix.png)