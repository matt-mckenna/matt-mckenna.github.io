---
layout: post
title: Training a next-word prediction model from scratch
---

In the last post we played around with BERT and saw that it predicted words pretty well. 
To show how much BERT improves word prediction over previous state-of-the-art models, we will train our own 
word prediction model using a lesser model.  The model we'll use is an LSTM - we're not going to delve into what 
these model are or how they work in the post (you can read about them [here](https://en.wikipedia.org/wiki/Long_short-term_memory))

The [full code](https://github.com/matt-mckenna/next_word_prediction) is much longer, but here the snippet that contains our model definition: 

```python 
    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=1))
        self.model.add(LSTM(self.lstm_units, return_sequences=True))
        self.model.add(LSTM(self.lstm_units))
        self.model.add(Dense(self.lstm_units, activation="relu"))
        self.model.add(Dense(self.vocab_size, activation="softmax"))
```

We can either train this model on input text file, or we can specify a wikipedia article to train on.  
For this example, we'll train on the wikipedia article about basketball. To do this, we'll specify the article and the numer of training epochs we want. 
We also input a phrase that we want the model to predict the next word for: "The NBA is a "

```python 
python modelpy.py --train_model --epochs 200 --wiki Basketball 
--predict "The NBA is a basketball"
```

And our model will predict the next word.  After training, we see the most likely next words are 'batter', 'ball', 'game'. The 'game' and 'ball' predictions make sense in the context of basketball. "Batter" is a strange choice and doesn't make sense. We can't be too hard on our model because it's trained on very little data and not trained for very long.  

Now lets compare to using BERT off the shelf (which doesn't require any training, as opposed to our LSTM model): For the same sentence of "The NBA is a basketball ___", BERT says the most likely words are: 'league', 'organization', 'competition'.
These make a lot more sense in context. 



