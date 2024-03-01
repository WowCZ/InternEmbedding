import os
import math
import time
import json
import tqdm
import torch
import random
import numpy as np
from typing import List
from accelerate import Accelerator
from embedding.train.loss import inbatch_negative_loss, hard_negative_loss
from embedding.train.training_embedder import initial_model, initial_opimizer_scheduler, get_train_dataloader

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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

def cache_gradients(query_embeds, passage_embeds, temperature):
    query_embeds = query_embeds.detach().requires_grad_()
    passage_embeds = passage_embeds.detach().requires_grad_()
    loss = inbatch_negative_loss(query_embeds, passage_embeds, temperature)
    loss.backward()

    return query_embeds.grad, passage_embeds.grad, loss.detach()

def train_embedder(args):
    setup_seed(args.seed) # 20
    args_dcit = vars(args)
    del args_dcit['func']

    args.ckpt_saving_dir = os.path.join(args.ckpt_saving_dir, f'{args.embedder_name}_{str(time.strftime("%Y%m%d%H%M%S", time.localtime()))}')
    os.makedirs(args.ckpt_saving_dir, exist_ok=True)

    with open(os.path.join(args.ckpt_saving_dir, 'embedder_training_cfg.json'), 'w') as fw:
        json.dump(args_dcit, fw, indent=4)

    accelerator = Accelerator(log_with=args.record_log)
    if args.record_log == 'wandb':
        accelerator.init_trackers(
            init_kwargs={"wandb":{"dir": args.ckpt_saving_dir}},
            project_name=args.wandb_project_name, 
            config={"learning_rate": args.learning_rate, "epochs": args.num_epochs, "batch_size": args.batch_size_per_gpu}
        )

    embedder = initial_model(args)
    train_loader = get_train_dataloader(args)

    if args.embedder_ckpt_path and os.path.exists(args.embedder_ckpt_path):
        print(f'>>> initial the model with {args.embedder_ckpt_path}')
        accelerator.wait_for_everyone()
        unwrapped_embedder = accelerator.unwrap_model(embedder)
        unwrapped_embedder.load_state_dict(torch.load(args.embedder_ckpt_path, map_location=torch.device(accelerator.device)))

    num_training_steps = (args.num_epochs * len(train_loader))
    optimizer, lr_scheduler = initial_opimizer_scheduler(args, embedder, num_training_steps)
    num_training_steps = math.ceil(num_training_steps / accelerator.num_processes)

    if accelerator.is_main_process:
        accelerator.print('#'*10, f' Embedder Training Config ', '#'*10)
        accelerator.print(json.dumps(args_dcit, indent=4))
        accelerator.print('#'*10, f' Training Dataset Statistics ', '#'*10)
        accelerator.print(train_loader.dataset)

        if args.peft_lora:
            embedder.encoder.backbone.print_trainable_parameters()

    # distributed training
    embedder, optimizer, train_loader, lr_scheduler = accelerator.prepare(embedder, optimizer, train_loader, lr_scheduler)
    accelerator.print(embedder)

    progress_bar = tqdm.tqdm(range(num_training_steps))

    embedder.train()
    train_step = 0
    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            accelerator.print('#'*10, f' Epoch {epoch} Starting ', '#'*10)
        for pq in train_loader:
            q_inputs, q_attention_mask, p_inputs, p_attention_mask, n_inputs, n_attention_mask = (item.to(accelerator.device) if item is not None else None for item in pq)

            if args.gradcache_chunk_size < 1:
                q_embeddings, p_embeddings, n_embeddings = embedder(q_inputs, q_attention_mask, p_inputs, p_attention_mask, n_inputs, n_attention_mask)
                loss = 0
                for matryoshka_dim in args.matryoshka_adaptive_dims:
                    matryoshka_selected_ids = list(range(matryoshka_dim))
                    matryoshka_q_embeddings, matryoshka_p_embeddings, matryoshka_n_embeddings = select_matryoshka_embedding([q_embeddings, p_embeddings, n_embeddings], matryoshka_selected_ids)
                    if args.hard_negative_sampling:
                        if n_embeddings is None:
                            raise RuntimeError('Negative embedding is None!')
                        
                        loss += hard_negative_loss(matryoshka_q_embeddings, matryoshka_p_embeddings, matryoshka_n_embeddings, args.temperature)
                    else:
                        loss += inbatch_negative_loss(matryoshka_q_embeddings, matryoshka_p_embeddings, args.temperature)

                accelerator.backward(loss)
            else:
                # Refer to https://github.com/luyug/GradCache/tree/main & https://github.com/nomic-ai/contrastors/blob/main/src/contrastors/loss.py
                # Step 1: Iterated inference process to get query and passage embeddings for each GradCache chunk
                query_embeds = []
                passage_embeds = []
                with torch.no_grad():
                    for chunk_left in range(0, args.batch_size_per_gpu, args.gradcache_chunk_size):
                        chunk_right = args.gradcache_chunk_size + chunk_left
                        q_chunk_embeds = embedder.embedding(q_inputs[chunk_left: chunk_right], q_attention_mask[chunk_left: chunk_right])
                        p_chunk_embeds = embedder.embedding(p_inputs[chunk_left: chunk_right], p_attention_mask[chunk_left: chunk_right])
                        query_embeds.append(q_chunk_embeds)
                        passage_embeds.append(p_chunk_embeds)

                    query_embeds = torch.concat(query_embeds, dim=0)
                    passage_embeds = torch.concat(passage_embeds, dim=0)

                # Step 2: Calculate the cached gradients of representation with inbatch negative loss
                query_grad_cache, passage_grad_cache, loss = cache_gradients(query_embeds, passage_embeds, args.temperature)
                
                # Step 3: Backpropagate the gradients according to the cached representation gradients
                query_grad_cache = query_grad_cache.split(args.gradcache_chunk_size)
                passage_grad_cache = passage_grad_cache.split(args.gradcache_chunk_size)
                for ci, chunk_left in enumerate(range(0, args.batch_size_per_gpu, args.gradcache_chunk_size)):
                    chunk_right = args.gradcache_chunk_size + chunk_left
   
                    q_chunk_embeds = embedder.embedding(q_inputs[chunk_left: chunk_right], q_attention_mask[chunk_left: chunk_right])
                    p_chunk_embeds = embedder.embedding(p_inputs[chunk_left: chunk_right], p_attention_mask[chunk_left: chunk_right])

                    q_chunk_cache = query_grad_cache[ci]
                    p_chunk_cache = passage_grad_cache[ci]

                    qp_chunk_embeds = torch.concat([q_chunk_embeds, p_chunk_embeds], dim=1)
                    qp_chunk_cache = torch.concat([q_chunk_cache, p_chunk_cache], dim=1)

                    surrogate = torch.dot(qp_chunk_embeds.flatten(), qp_chunk_cache.flatten())
                    accelerator.backward(surrogate)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                train_step += 1
                cur_lr = lr_scheduler.get_lr()[0]

                if train_step % args.save_ckpt_steps == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_embedder = accelerator.unwrap_model(embedder)

                    if args.only_last_ckpt and accelerator.is_main_process:
                        last_ckpt_file = os.path.join(args.ckpt_saving_dir, f"{args.embedder_name}_{train_step-args.save_ckpt_steps}.pt")
                        if os.path.exists(last_ckpt_file):
                            os.remove(last_ckpt_file)

                    accelerator.print('>>> save the checkpoint into ', args.ckpt_saving_dir)
                    accelerator.save(unwrapped_embedder.state_dict(), os.path.join(args.ckpt_saving_dir, f"{args.embedder_name}_{train_step}.pt"))

            accelerator.log(dict(step=train_step, loss=loss, learning_rate=float(cur_lr)), step=train_step)

    accelerator.wait_for_everyone()
    unwrapped_embedder = accelerator.unwrap_model(embedder)
    accelerator.print('>>> save the checkpoint into ', args.ckpt_saving_dir)
    accelerator.save(unwrapped_embedder.state_dict(), os.path.join(args.ckpt_saving_dir, f"{args.embedder_name}_{train_step}.pt"))