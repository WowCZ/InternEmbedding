
import os
import math
import torch
from transformers import AutoTokenizer, get_scheduler
from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.data.data_loader import train_dataloader
from embedding.data.datasets import EmbedderDatasets
from embedding.data.data_utils import training_datatset_files

def initial_model(args):
    if args.backbone_type == 'BERT':
        embedder = BertEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=args.checkpoint_batch_size)

    elif args.backbone_type == 'Mistral':
        # TODO: when pool_type is eos, the loss is NaN
        mytryoshka_indexes = list(range(args.mytryoshka_size))
        embedder = MistralEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=args.checkpoint_batch_size, lora_config=args.peft_lora, which_layer=args.which_layer, mytryoshka_indexes=mytryoshka_indexes)
    
    else:
        raise TypeError(f'The type of backbone {args.backbone_type} has not been supported yet!')

    return embedder

def initial_opimizer_scheduler(args, embedder, num_training_steps):
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        args.schedule_type, # "linear"
        optimizer=optimizer,
        num_warmup_steps=math.ceil(num_training_steps*args.warmup_rate), # 0.01
        num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler

def get_train_dataloader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.init_backbone)
    if args.backbone_type in ['Mistral']:
        # uncomment: when padding token is not set, like Mistral 
        tokenizer.pad_token = tokenizer.eos_token

    paired_embedding_dataset = EmbedderDatasets(training_datatset_files, task_prompt=args.task_prompt)
    train_loader = train_dataloader(paired_embedding_dataset, tokenizer, max_length=args.max_length, sampler='random', batch_size=args.batch_size_per_gpu)
    return train_loader

def get_eval_dataloader():
    pass