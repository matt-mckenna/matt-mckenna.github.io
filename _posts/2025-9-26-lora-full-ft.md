---
layout: post
title: LoRA vs. Full Parameter Fine-Tuning
---

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
