
import math
import torch
from embedding.models import MODEL_MAPPING, PREF_MODEL_MAPPING
from transformers import AutoTokenizer, get_scheduler
from embedding.data.data_loader import train_dataloader
from embedding.data.datasets import EmbedderDatasets, EmbedderConcatDataset

def initial_model(args):
    # mytryoshka_indexes = list(range(args.mytryoshka_size))
    tokenizer = AutoTokenizer.from_pretrained(args.init_backbone, trust_remote_code=True)
    if args.backbone_type in ['Mistral']:
        # uncomment: when padding token is not set, like Mistral 
        tokenizer.pad_token = tokenizer.eos_token

    if args.backbone_type in PREF_MODEL_MAPPING:
        embedder = PREF_MODEL_MAPPING[args.backbone_type](
            args.init_backbone, 
            pool_type=args.pool_type, 
            checkpoint_batch_size=args.checkpoint_batch_size, 
            lora_config=args.peft_lora, 
            which_layer=-1, 
            mytryoshka_indexes=None, 
            normalize=False)
    else:
        raise TypeError(f'The type of backbone {args.backbone_type} has not been supported yet!')
    return embedder, tokenizer

def initial_opimizer_scheduler(args, embedder, num_training_steps):
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        args.schedule_type, # "linear"
        optimizer=optimizer,
        num_warmup_steps=math.ceil(num_training_steps*args.warmup_rate), # 0.01
        num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler

def get_train_dataloader(args, tokenizer):
    if args.sampler in ['random']:
        print(f'Overload the EmbedderDatasets class with the preference dataset class')
        embedding_dataset = EmbedderDatasets(args.dataset_config, 
                                             task_prompt=args.task_prompt, 
                                             negative_num=args.hard_negative_num,
                                             args=args)
    else:
        raise NotADirectoryError(f'>>> The sampler {args.sampler} has not been supported yet!')

    train_loader = train_dataloader(embedding_dataset, 
                                    tokenizer, 
                                    max_length=args.max_length, 
                                    sampler=args.sampler, 
                                    batch_size=args.batch_size_per_gpu)
    
    return train_loader

def get_eval_dataloader(args, tokenizer):
    embedding_dataset = EmbedderDatasets(args.dataset_config,
                                         task_prompt=False,
                                         negative_num=0,
                                         file_name='eval.jsonl' if args.dev_mode else 'test.jsonl',
                                         args=args)
    eval_loader = train_dataloader(embedding_dataset,
                                   tokenizer,
                                   max_length=args.max_length,
                                   sampler='sequential',
                                   batch_size=args.batch_size_per_gpu)
    return eval_loader
