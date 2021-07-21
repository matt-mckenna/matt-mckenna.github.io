---
layout: post
title: Using pre-trained BERT to predict words in a sentence
---

BERT is a language model that has shown state of the art results on many natural language tasks ([see here for a more in-depth explanation](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)). 
BERT works by masking certain words in text, then trying to 'recover' those masked words. 
For example, in the sentence "The cat ate the mouse", BERT might mask the word 'mouse', 
then try to predict that word in the sentence "The cat ate the [MASK]". 
We'll go more in-depth on what BERT is and how it works in later posts - in this post
we'll play around with BERT and see how we can use it to predict words in a sentence. 
To do this, we'll follow these steps:

1. Load the transformer library from Huggingface
2. Tokenize our input sentence (convert words to integers) 
3. Run the tokenized sentence through the model 
4. Find the top 'k' words predicted for our target word 
5. Decode the tokens back into words (convert integers to words) 
     

This is surprisingly simple with the Huggingface (transformers) library.  
We'll use that library to write a function that performs the steps above. 
    
```python 
import numpy as np
from transformers import BertTokenizer, TFBertForMaskedLM
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForMaskedLM.from_pretrained('bert-base-cased')

def get_top_k_predictions(input_string, k=5, tokenizer=tokenizer, model=model) -> str:

    tokenized_inputs = tokenizer(input_string, return_tensors="tf")
    outputs = model(tokenized_inputs["input_ids"])

    top_k_indices = tf.math.top_k(outputs.logits, k).indices[0].numpy()
    decoded_output = tokenizer.batch_decode(top_k_indices)
    mask_token = tokenizer.encode(tokenizer.mask_token)[1:-1]
    mask_index = np.where(tokenized_inputs['input_ids'].numpy()[0]==mask_token)[0][0]

    decoded_output_words = decoded_output[mask_index]

    return decoded_output_words

```

We can use our function to predict the masked word in any sentence.  We 
send in a sentence with a '[MASK]' placeholder for the word that we want BERT to predict, 
and the function will output the top 5 most likely words (from most likely to least likely). 
Here are a few examples:

    get_top_k_predictions("The dog ate the [MASK]. ")
    output:  'food meat dog bread meal'
    
    get_top_k_predictions("The capital of France is [MASK]. ")
    output:  'Paris Lyon Toulouse Lille Marseille'

    get_top_k_predictions("The boy played with the [MASK] at the pool. ")
    output: 'girl boy fish girls woman'
    
    get_top_k_predictions("The Boston [MASK] won the championship. ")
    output: 'Celtics Bruins Braves Patriots Americans'
    
As you can see, BERT does pretty well predicting what the [MASK]'ed word should be!
    