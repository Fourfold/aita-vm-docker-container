---
library_name: peft
license: other
base_model: ModelSpace/GemmaX2-28-9B-Pretrain
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: gemmax2_9b_finetuned_v5
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# gemmax2_9b_finetuned_v5

This model is a fine-tuned version of [ModelSpace/GemmaX2-28-9B-Pretrain](https://huggingface.co/ModelSpace/GemmaX2-28-9B-Pretrain) on the training_set dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 3
- total_train_batch_size: 24
- total_eval_batch_size: 64
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.6.0+cu124
- Datasets 3.6.0
- Tokenizers 0.21.1