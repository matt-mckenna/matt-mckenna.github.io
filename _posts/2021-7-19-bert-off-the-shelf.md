---
layout: post
title: Using pre-trained BERT to predict words in a sentence
---

BERT is a language model that has shown state of the art results on many natural language tasks ([see here for a more in-depth explanation](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)). BERT works by masking words in text, then trying to 'recover' what the masked word is. For example, In this post, we'll play around with BERT and see how we can use it to predict words in a sentence. To do this, we'll use Huggingface's transfomer library. 
    
First, we can load some 
    import numpy as np
    from transformers import BertTokenizer, TFBertForMaskedLM
    import tensorflow as tf
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForMaskedLM.from_pretrained('bert-base-cased')
    
    def get_top_k_predictions(input_string, k, tokenizer=tokenizer, model=model) -> str:
    
        tokenized_inputs = tokenizer(input_string, return_tensors="tf")
        outputs = model(tokenized_inputs["input_ids"])
    
        top_k_indices = tf.math.top_k(outputs.logits, k).indices[0].numpy()
        decoded_output = tokenizer.batch_decode(top_k_indices)
        mask_token = tokenizer.encode(tokenizer.mask_token)[1:-1]
        mask_index = np.where(tokenized_inputs['input_ids'].numpy()[0]==mask_token)[0][0]
    
        decoded_output_words = decoded_output[mask_index]
    
        return decoded_output_words

We

    get_top_k_predictions("The [MASK] are the NBA team from Boston.", 5) 