import os
import math
import tqdm
import torch
import random
import numpy as np
from typing import List
from accelerate import Accelerator
from transformers import AutoTokenizer, get_scheduler
from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.data.data_loader import train_dataloader
from embedding.data.datasets import EmbedderDatasets
from embedding.train.loss import inbatch_negative_loss, hard_negative_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)

accelerator = Accelerator(log_with='wandb')
which_layer = -1
batch_size = 300 #300
num_epochs = 1
max_length = 512
checkpoint_batch_size = 10
save_ckpt_steps = 500
only_last_ckpt = False
hard_negative_sampling = False
temperature = 0.02 # 0.02
learning_rate = 2e-4
matryoshka_adaptive_dims = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
accelerator.init_trackers(
    init_kwargs={"wandb":{"dir": "/fs-computility/llm/chenzhi/wandb"}},
    project_name="MistralEmbedder", 
    config={"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size}
)

# wandb_tracker = accelerator.get_tracker("wandb")
# if accelerator.is_main_process:
#     wandb_tracker.log_artifact(dir='/fs-computility/llm/chenzhi/wandb')

# initial model
peft_lora = True
# backbone = "google/bert_uncased_L-8_H-512_A-8"
# embedder = BertEmbedder(backbone, pool_type='mean', checkpoint_batch_size=64)
backbone = "/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e"
# TODO: when pool_type is eos, the loss is NaN
embedder = MistralEmbedder(backbone, pool_type='position_weight', checkpoint_batch_size=checkpoint_batch_size, lora_config=True, which_layer=which_layer, mytryoshka_indexes=list(range(4096)))
if peft_lora and accelerator.is_main_process:
    embedder.encoder.backbone.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(backbone)
# uncomment: when padding token is not set, like Mistral 
tokenizer.pad_token = tokenizer.eos_token

# initial dataloader
## all samples are randomly sampled, where one batch contains different domain samples. 
# datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/ELI5/train.jsonl', 
#                   '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
#                   '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl']

## all samples in a batch are sampled from the same task, which we called the in-domain batch sampling. 
# datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/InDomain/train.jsonl']

##  msmarco dataset with hard negative sampling
# datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/MSMARCO_Triple/train.jsonl']

# datatset_files = [
#             '/fs-computility/llm/chenzhi/datasets_processed/ELI5/filtered_phase2_train.jsonl', 
#             '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MultiNLI/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/Quora/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/filtered_phase2_train.jsonl',
#             '/fs-computility/llm/chenzhi/datasets_processed/FEVER/filtered_phase2_train.jsonl'
# ]

datatset_files = ['/fs-computility/llm/chenzhi/datasets_processed/STELI5/train.jsonl', 
                  '/fs-computility/llm/chenzhi/datasets_processed/HotpotQA/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/MSMARCO/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/STAllNLI/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/Quora/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/MIRACL/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/MrTyDi/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/SQuAD/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/NautralQuestions/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/TriviaQA/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/FEVER/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/DuReader/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/T2Ranking/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/STGooQA/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/STSpecter/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/STStackexchangeDup/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/STWikiHow/train.jsonl',
                  '/fs-computility/llm/chenzhi/datasets_processed/STYahooQA/train.jsonl']

eli5_dataset = EmbedderDatasets(datatset_files, task_prompt=True)
train_loader = train_dataloader(eli5_dataset, tokenizer, max_length=max_length, sampler='random', batch_size=batch_size)

# initial trainer
optimizer = torch.optim.AdamW(embedder.parameters(), lr=learning_rate)

num_training_steps = (num_epochs * len(train_loader))
if accelerator.is_main_process:
    accelerator.print('>>> num_training_steps: ', num_training_steps)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=math.ceil(num_training_steps*0.01),
    num_training_steps=num_training_steps
)

def select_matryoshka_embedding(full_embeddings: List[torch.Tensor], select_ids: list):
    mytryoshka_embeddings = []
    for full_embedding in full_embeddings:
        if full_embedding is not None:
            device = full_embedding.device
            mytryoshka_embedding = full_embedding.index_select(-1, torch.tensor(select_ids).to(device))
        else:
            mytryoshka_embedding = None
        mytryoshka_embeddings.append(mytryoshka_embedding)
    return mytryoshka_embeddings

# distributed training
embedder, optimizer, train_loader, lr_scheduler = accelerator.prepare(embedder, optimizer, train_loader, lr_scheduler)
accelerator.print(embedder)

progress_bar = tqdm.tqdm(range(math.ceil(num_training_steps / accelerator.num_processes)))

embedder.train()
train_step = 0
for epoch in range(num_epochs):
    for pq in train_loader:
        q_inputs, q_attention_mask, p_inputs, p_attention_mask, n_inputs, n_attention_mask = (item.to(accelerator.device) if item is not None else None for item in pq)
        q_embeddings, p_embeddings, n_embeddings = embedder(q_inputs, q_attention_mask, p_inputs, p_attention_mask, n_inputs, n_attention_mask)
        loss = 0
        for matryoshka_dim in matryoshka_adaptive_dims:
            matryoshka_selected_ids = list(range(matryoshka_dim))
            matryoshka_q_embeddings, matryoshka_p_embeddings, matryoshka_n_embeddings = select_matryoshka_embedding([q_embeddings, p_embeddings, n_embeddings], matryoshka_selected_ids)
            if n_embeddings is not None and hard_negative_sampling:
                loss += hard_negative_loss(matryoshka_q_embeddings, matryoshka_p_embeddings, matryoshka_n_embeddings, temperature)
            else:
                loss += inbatch_negative_loss(matryoshka_q_embeddings, matryoshka_p_embeddings, temperature)

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            progress_bar.update(1)
            train_step += 1
            cur_lr = lr_scheduler.get_lr()[0]

            if train_step % save_ckpt_steps == 0:
                accelerator.wait_for_everyone()
                unwrapped_embedder = accelerator.unwrap_model(embedder)

                if only_last_ckpt and accelerator.is_main_process:
                    last_ckpt_file = os.path.join("/fs-computility/llm/chenzhi/ckpts", f"mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs{batch_size}_ml{max_length}_{train_step-save_ckpt_steps}.pt")
                    if os.path.exists(last_ckpt_file):
                        os.remove(last_ckpt_file)

                accelerator.save(unwrapped_embedder.state_dict(), os.path.join("/fs-computility/llm/chenzhi/ckpts", f"mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs{batch_size}_ml{max_length}_{train_step}.pt"))

        accelerator.log(dict(step=train_step, loss=loss, learning_rate=float(cur_lr)), step=train_step)

accelerator.wait_for_everyone()
unwrapped_embedder = accelerator.unwrap_model(embedder)
accelerator.save(unwrapped_embedder.state_dict(), os.path.join("/fs-computility/llm/chenzhi/ckpts", f"mistral_embedder18_full_pow1_prompt_ins_matryoshka9_temp2_lr24_bs{batch_size}_ml{max_length}_{train_step}.pt"))