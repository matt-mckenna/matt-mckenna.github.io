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

$$ B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k} $$

so we set $ ∆W = BA $ and the forward pass is 

$$ h = W_0x + ∆Wx = W_0x + BAx $$

$ W_0x $ is also scaled by $ \alpha / r $. The authors say "$\alpha$ is a constant in $r$" which means $\alpha$ does not change if you change the rank of $r$. Below you'll see that you can specify $\alpha$ and $r$ in code when you use LoRA. 

## How to apply LoRA?

One consideration that's not always covered when talking about LoRA is *where* you should apply LoRA. In transformer models you have many different weight matrices - which ones should you apply LoRA to and why? Since LoRA is used for fine-tuning, you're usually aiming to change the model behavior in some way (without destroying all the knowledge the model learned from pre-training). In that sense you want to taget matrics that change model behavior: the attention projections (Q, K, V, O). Many times practitioners apply LoRA to the attention weights first, then other weights if needed. 


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

but later (about 4 years after the original LoRA paper came out!) John Schulman from Thinking Machines [wrote an amazing blog post](https://thinkingmachines.ai/blog/lora/) showing why you should actually target the MLP layers in addition to the attention layers.  

> Even in small data settings, LoRA performs better when applied to all weight matrices, 
> especially MLP and MoE layers. Attention-only LoRA underperforms even when we match the number 
> of trainable parameters by using higher rank for attention-only LoRA.

## When to look at LoRA vs. standard fine-tuning? 

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
