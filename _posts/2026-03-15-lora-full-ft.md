---
layout: post
title: LoRA vs Full Fine-Tuning
date: 2026-03-15
---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter efficient method to fine-tune LLMs for particular tasks or domains. The main benefit of LoRA is to reduce the amount compute/memory needed by only training on a small portion (usually less than 1%!) of the original model parameters. There are other major benefits of LoRA too: 

 - Using LoRA doesn't add any inference latency over a fine-tuned model 
 - LoRA weights can be swapped for a shared base model. You can have one copy of a base model (BERT, Qwen, etc.) in memory, and swap LoRA adaptors when you need to use the model for different tasks. This can massively reduce storage overhead and efftively lets you have one model that can perform multiple tasks. 

In short, LoRA achieves task-specific adaptation by learning low-rank updates on top of a frozen base model, delivering fine-tuning quality with dramatically lower training, memory, and deployment costs.

## How does LoRA work?

The key to LoRA comes from the fact that LLMs have a low "intrinsic rank", which means they still learn well in lower dimensions/# of parameters. This was demonstrated in the 2020 paper [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/pdf/2012.13255):

> by optimizing only 200 trainable parameters randomly projected back into the full space, we
> can tune a RoBERTa model to achieve 90% of the full parameter performance levels

The key insight with LoRA is that you don't need to re-train all the parameters in the model in order to see good performance on new tasks. With LoRA, we freeze (i.e. force to stay constant) most of the model weights and only update a certain subset of new weights. That's why LoRA is much more efficient to train. 

In full fine-tuning, the weight update is: 

$$ W = W_0 + ∆W $$

In LoRA, the model weight update becomes: 

$$ W = W_0 + BA $$
The trick is the dimensionality of ∆W is much smaller than BA.

In this setup,

$ W_0 $ is the weight matrix of the original model and 

$$ B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k} $$

so we set $ ∆W = BA $ and the forward pass is 

$$ h = W_0x + ∆Wx = W_0x + BAx $$

$ W_0x $ is also scaled by $ \alpha / r $. The authors say "$\alpha$ is a constant in $r$" which means $\alpha$ does not change if you change the rank of $r$. Below you'll see that you can specify $\alpha$ and $r$ in code when you use LoRA. 


## How to apply LoRA

One consideration that's not always covered when talking about LoRA is *where* in the model (i.e. to what weight matrices) you should apply LoRA. In transformer models you have many different weight matrices - which ones should you apply LoRA to and why? Since LoRA is used for fine-tuning, you're usually aiming to change the model behavior in some way (without destroying all the knowledge the model learned from pre-training). In that sense you want to taget matrics that change model behavior: the attention projections (Q, K, V, O). A common approach is to apply LoRA to the attention weights first, then other weights if needed. 

### LoRA in code

The easiest way to apply LoRA to LLM is using PyTorch + Huggingface's transformers library. First we load a model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "meta-llama/Llama-2-7b-hf"  # example

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

Then configure LoRA

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # rank
    lora_alpha=16,            # scaling
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # attention layers to target
    bias="none",
)
```

Now apply LoRA to your model!

```python
model = get_peft_model(model, lora_config)
```

if you print the trainable parameters, you'll see something like: 

```python
trainable params: 4.2M || all params: 6.7B || trainable%: 0.06%
```

so LoRA is only training 0.06% of the original parameters! That's huge and translates into massive compute/memory efficiency.

One intresting note is about which parameters to train. The authors of the LoRA paper say: 

> We limit our study to only adapting the attention weights for downstream
> tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity
> and parameter-efficiency

but later (about 4 years after the original LoRA paper came out!) John Schulman from Thinking Machines [wrote an amazing blog post](https://thinkingmachines.ai/blog/lora/) showing why you should target the MLP layers in addition to the attention layers.  

> Even in small data settings, LoRA performs better when applied to all weight matrices, 
> especially MLP and MoE layers. Attention-only LoRA underperforms even when we match the number 
> of trainable parameters by using higher rank for attention-only LoRA.

## LoRA vs. full fine-tuning

Both LoRA and full fine-tuning are ways to fine-tune LLMs. When and why should we pick one over the other? Above we saw that LoRA is much more efficient than full fine-tuning, so why not always use LoRA? Like most things in life, it's a tradeoff. Let's look at the pros and cons of both: 

Full fine-tuning is good for:
- tasks requiring high accuracy and task-specific understanding (legal or financial document analysis)
- specialized vocabulary or complex subject matter, e.g. medicine, law, finance
- comprehensive adaptation to the new data
- You want the model to forget pretraining quirks (toxicity, bias)
- You plan to ship a single high-quality model, not many variants of that model (adapters)

LoRA is good for
- Lower resources (GPUs)
- You’re adapting a foundation model to a narrow task (e.g., sentiment classification, SQL translation).
- The task is somewhat generic and an existing LLM can perform well
- You have multiple taks you want to perform with the same base model

LoRA is still popular for good reason — but in practice the tradeoffs aren't always obvious. Let's look at an experiment.

## Experiment: LoRA vs Full Fine-Tuning on PubMedQA

To make this concrete, I ran both methods on a real medical QA task and the MMLU benchmark. This is a small experient to show the directional difference of LoRA and full FT. I wanted to see how LoRA and full FT compare on domain-specific data (medical data). The PubMedQA dataset measures how well LoRA and full FT can adapt to a new domain, the MMLU benchmark measures how much the FT affects generanl knowledge. The setup:

- **Model**: Qwen2.5-3B-Instruct
- **Training data**: PubMedQA (`pqa_labeled`, ~900 training examples)
- **Task**: 3-way classification — given a biomedical question and abstract, answer yes/no/maybe
- **General knowledge benchmark**: MMLU (1000 random samples) to measure forgetting
- **LoRA config**: rank=16, alpha=32, targeting all attention + MLP projections (see explanation above!)

### Results

<table style="border-collapse: collapse; width: 100%; margin: 1em 0;">
  <thead>
    <tr>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Method</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">PubMedQA</th>
      <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">MMLU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Base (no fine-tuning)</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.40</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.605</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">LoRA (10 epochs)</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.45</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.589</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">Full FT (10 epochs)</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.58</td>
      <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">0.583</td>
    </tr>
  </tbody>
</table>

### What this shows

**Full fine-tuning wins on domain adaptation.** Full FT reaches 0.58 on PubMedQA vs 0.45 for LoRA — a meaningful gap on a hard task with limited training data. Shows evidence that full FT can learn domain-specific data better. 

**LoRA preserves general knowledge better.** MMLU drops only 0.016 points with LoRA vs 0.022 with full FT. Small difference here, but it grows with more aggressive training. Shows evidence that full FT has more forgetting (although very slight difference here). 


### Wrap-up

LoRA is a useful way to efficiently fine-tune models. You can get comparable results with full FT, but LoRA has drawbacks too - mostly that it can underperform full FT in certain situations, like when the data you're tuning on is different that what the base models were trained on. But for most practical tasks, LoRA is a good choice that can get you far.

Code for this experiment is available [on GitHub](https://github.com/matt-mckenna/blog-examples/tree/main/lora-vs-fullft).