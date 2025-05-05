---
layout: post
title: Predicting Masked Words with BERT (Refreshed for 2025)
---

> **Note (May 2025):** This post is an updated version of my original 2020 walkthrough of using BERT for masked word prediction. I’ve refreshed the code to use the latest version of Hugging Face Transformers with PyTorch instead of TensorFlow, and clarified the explanation and examples.

BERT is a foundational language model that changed the way we approach many NLP tasks. One of its core training objectives is *masked language modeling*—predicting missing words in a sentence.

For example, given:
> “The cat ate the [MASK].”

BERT tries to recover the missing word, e.g., “mouse” or “food.”

Let’s walk through how to use `bert-base-cased` from Hugging Face to get BERT’s top predictions for a masked word.

---

### 🔧 Steps

1. Load the model and tokenizer  
2. Add a `[MASK]` token to your sentence  
3. Run inference  
4. Decode the top-k predicted tokens

---

### 🧪 Code

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertForMaskedLM.from_pretrained("bert-base-cased")
model.eval()  # Set to eval mode

def get_top_k_predictions(text, k=5):
    # Tokenize input with masking
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]

    # Get top k tokens
    top_k_tokens = torch.topk(mask_token_logits, k, dim=1).indices[0].tolist()
    return tokenizer.convert_ids_to_tokens(top_k_tokens)

```

Some examples:

```python
get_top_k_predictions("The dog ate the [MASK].")
# → ['food', 'meat', 'dog', 'bread', 'meal']

get_top_k_predictions("The capital of France is [MASK].")
# → ['Paris', 'Lyon', 'Toulouse', 'Marseille', 'Nice']

get_top_k_predictions("The Boston [MASK] won the championship.")
# → ['Celtics', 'Bruins', 'Red', 'Patriots', 'Sox']
```