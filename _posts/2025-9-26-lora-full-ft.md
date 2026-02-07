---
layout: post
title: LoRA vs. Full Parameter Fine-Tuning
---

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter efficient method to fine-tune LLMs for particular tasks or domains. The main benefit of LoRA is to reduce the amount compute/memory needed by only training on a small portion of the model parameters. There are other major benefits of LoRA too! 

 - Using LoRA doesn't add any inference latency over a fine-tuned model 
 - LoRA weights can be swapped for a shared base model. So you can have one copy of a base model (BERT, Qwen, etc.) in memory, and swap LoRA adaptors when you need to use the model for different tasks. This can massively reduce storage overhead.  

## How does LoRA work?

The key to LoRA comes from the fact that LLMs have a low "intrinsic rank", which means they still learn well in lower dimensions/# of parameters (more formally, they). This was demonstrated in the 2020 paper [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/pdf/2012.13255):

> by optimizing only 200 trainable parameters randomly projected back into the full space, we
> can tune a RoBERTa model to achieve 90% of the full parameter performance levels

With LoRA, the model weight update becomes: 

$$ W_0 + ∆W = W_0 + BA $$

where 

$ W_0 $ is the weight matrix of the original model and 
cd
$B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}$

so we set $ ∆W = BA $ and the forward pass is 

$$ h = W_0x + ∆Wx = W_0x + BAx $$

## When to look at LoRA vs. fine-tuning? 

Both LoRA (and more generally PEFT) are ways to fine-tune LLMs. When and why should we pick one over the other? First let’s understand the differences. 

### Full fine-tuning
- Update all parameters of the model.
- Requires lots of GPU memory, long training runs, and careful optimization.
- Produces a single “frozen” model per task.


### LoRA (Low-Rank Adapters)
- Freeze the base model; inject trainable low-rank matrices into attention layers.
- Train far fewer parameters (often <1%).
- Can load/swap adapters on the fly.
- Nearly identical inference cost to the base model.
- These differences matter because they affect:
- Cost (GPU hours, memory)
- Speed (training + inference latency)
- Flexibility (can you swap adapters per task?)
- Performance (final accuracy or quality)


Full parameter fine tuning is good for:
- tasks requiring high accuracy and task-specific understanding (legal or financial document analysis)
- specialized vocabulary or complex subject matter, like medicine, law, or finance
- comprehensive adaptation to the new data
- You want the model to forget pretraining quirks (e.g., toxicity, bias)
- You plan to ship a single high-quality model, not many variants of that model (adapters)

LoRA is good for
- When you have lower resources (GPUs)
- You’re adapting a foundation model to a narrow task (e.g., sentiment classification, SQL translation).
- The task is somewhat generic and an existing LLM can perform well
