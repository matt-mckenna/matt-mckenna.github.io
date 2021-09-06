---
layout: post
title: Training a next-word prediction model from scratch
---

In the last post we played around with BERT and saw that it predicted words pretty well. 
To show how much BERT improves word prediction over previous state-of-the-art models, we will train our own 
word prediction model using a lesser model.  The model we'll use is an LSTM - we're not going to delve into what 
these model are or how they work in the post (you can read about them [here](https://en.wikipedia.org/wiki/Long_short-term_memory))




```python 
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 10, input_length=1))
        self.model.add(LSTM(10, return_sequences=True))
        self.model.add(LSTM(self.lstm_units))
        self.model.add(Dense(1000, activation="relu"))
        self.model.add(Dense(self.vocab_size, activation="softmax")) 
```

