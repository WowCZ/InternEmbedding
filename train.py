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
from accelerate.state import AcceleratorState
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

def cache_gradients(query_embeds, passage_embeds, negative_embeds, temperature, task_adaptation, task_type_inbatch, hard_negative: bool=True):
    query_embeds = query_embeds.detach().requires_grad_()
    passage_embeds = passage_embeds.detach().requires_grad_()

    if negative_embeds is not None:
        negative_embeds = negative_embeds.detach().requires_grad_()

    loss = 0
    if not task_adaptation or task_type_inbatch not in ['Clustering', 'Classification']:
        loss = inbatch_negative_loss(query_embeds, passage_embeds, temperature)

    if task_adaptation and not hard_negative and negative_embeds is None:
        raise ValueError('The hard negative sampling should be adopted, when task adaptation mode is open. \
                            Otherewise, the training process will miss Clustering and Classification datasets')

    if negative_embeds is not None and hard_negative:
        loss += hard_negative_loss(query_embeds, passage_embeds, negative_embeds, temperature)

    loss.backward()

    negative_embeds_grad = None
    if negative_embeds is not None and hard_negative:
        negative_embeds_grad = negative_embeds.grad        

    return query_embeds.grad, passage_embeds.grad, negative_embeds_grad, loss.detach()

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

    embedder, tokenizer = initial_model(args)
    train_loader = get_train_dataloader(args, tokenizer)

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
    if accelerator.distributed_type == 'DEEPSPEED':
        AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] =  args.batch_size_per_gpu
    
    embedder, optimizer, train_loader, lr_scheduler = accelerator.prepare(embedder, optimizer, train_loader, lr_scheduler)
    accelerator.print(embedder)

    progress_bar = tqdm.tqdm(range(num_training_steps))

    # TODO: negative sampling with margin scale refer to https://gombru.github.io/2019/04/03/ranking_loss/
    embedder.train()
    train_step = 0
    if accelerator.is_main_process:
        if args.hard_negative_sampling:
            lfunc = 'hard_negative_loss'
        else:
            lfunc = 'inbatch_negative_loss'
        accelerator.print(f'>>> Training loss function is {lfunc}')
        
    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            accelerator.print('#'*10, f' Epoch {epoch} Starting ', '#'*10)
        for pq in train_loader:
            q_inputs, p_inputs, n_list_inputs, task_type_inbatch = pq
            q_inputs = dict([(k, v.to(accelerator.device)) for k, v in q_inputs.items()])
            p_inputs = dict([(k, v.to(accelerator.device)) for k, v in p_inputs.items()])
            if n_list_inputs:
                n_list_inputs = [dict([(k, v.to(accelerator.device)) for k, v in n_inputs.items()]) for n_inputs in n_list_inputs]

            if args.gradcache_chunk_size < 1:
                q_embeddings, p_embeddings, n_embeddings = embedder(q_inputs, p_inputs, n_list_inputs)
                loss = 0
                for matryoshka_dim in args.matryoshka_adaptive_dims:
                    matryoshka_selected_ids = list(range(matryoshka_dim))
                    matryoshka_q_embeddings, matryoshka_p_embeddings, matryoshka_n_embeddings = select_matryoshka_embedding([q_embeddings, p_embeddings, n_embeddings], matryoshka_selected_ids)
                    
                    if not args.task_adaptation or task_type_inbatch not in ['Clustering', 'Classification']:
                        loss += inbatch_negative_loss(matryoshka_q_embeddings, matryoshka_p_embeddings, args.temperature)

                    if args.task_adaptation and not args.hard_negative_sampling and n_embeddings is None:
                        raise ValueError('The hard negative sampling should be adopted, when task adaptation mode is open. \
                                         Otherewise, the training process will miss Clustering and Classification datasets.')

                    if n_embeddings and args.hard_negative_sampling:
                        loss += hard_negative_loss(matryoshka_q_embeddings, matryoshka_p_embeddings, matryoshka_n_embeddings, args.temperature)                        
                        
                accelerator.backward(loss)
            else:
                # Refer to https://github.com/luyug/GradCache/tree/main & https://github.com/nomic-ai/contrastors/blob/main/src/contrastors/loss.py
                # Step 1: Iterated inference process to get query and passage embeddings for each GradCache chunk
                query_embeds = []
                passage_embeds = []
                negative_embeds = []
                with torch.no_grad():
                    for chunk_left in range(0, args.batch_size_per_gpu, args.gradcache_chunk_size):
                        chunk_right = args.gradcache_chunk_size + chunk_left
                        q_chunk_inputs = dict([(k, v[chunk_left: chunk_right]) for k, v in q_inputs.items()])
                        p_chunk_inputs = dict([(k, v[chunk_left: chunk_right]) for k, v in p_inputs.items()])
                        n_chunk_inputs = None
                        if n_list_inputs:
                            n_chunk_inputs = [dict([(k, v[chunk_left: chunk_right]) for k, v in n_inputs.items()]) for n_inputs in n_list_inputs]
                        q_chunk_embeds, p_chunk_embeds, n_chunk_embeds = embedder(q_chunk_inputs, p_chunk_inputs, n_chunk_inputs)

                        query_embeds.append(q_chunk_embeds)
                        passage_embeds.append(p_chunk_embeds)
                        if n_chunk_embeds is not None:
                            negative_embeds.append(n_chunk_embeds)

                    query_embeds = torch.concat(query_embeds, dim=0)
                    passage_embeds = torch.concat(passage_embeds, dim=0)
                    if negative_embeds:
                        negative_embeds = torch.concat(negative_embeds, dim=0)
                    else:
                        negative_embeds = None

                # Step 2: Calculate the cached gradients of representation with inbatch negative loss
                query_grad_cache, passage_grad_cache, negative_grad_cache, loss = cache_gradients(query_embeds, passage_embeds, negative_embeds, args.temperature, args.task_adaptation, task_type_inbatch, args.hard_negative_sampling)
                
                # Step 3: Backpropagate the gradients according to the cached representation gradients
                query_grad_cache = query_grad_cache.split(args.gradcache_chunk_size)
                passage_grad_cache = passage_grad_cache.split(args.gradcache_chunk_size)
                if negative_grad_cache is not None:
                    negative_grad_cache = negative_grad_cache.split(args.gradcache_chunk_size)

                for ci, chunk_left in enumerate(range(0, args.batch_size_per_gpu, args.gradcache_chunk_size)):
                    chunk_right = args.gradcache_chunk_size + chunk_left
   
                    q_chunk_inputs = dict([(k, v[chunk_left: chunk_right]) for k, v in q_inputs.items()])
                    p_chunk_inputs = dict([(k, v[chunk_left: chunk_right]) for k, v in p_inputs.items()])
                    n_chunk_inputs = None
                    if n_list_inputs:
                        n_chunk_inputs = [dict([(k, v[chunk_left: chunk_right]) for k, v in n_inputs.items()]) for n_inputs in n_list_inputs]

                    q_chunk_embeds, p_chunk_embeds, n_chunk_embeds = embedder(q_chunk_inputs, p_chunk_inputs, n_chunk_inputs)

                    q_chunk_cache = query_grad_cache[ci]
                    p_chunk_cache = passage_grad_cache[ci]

                    n_chunk_cache = None
                    if negative_grad_cache:
                        n_chunk_cache = negative_grad_cache[ci]                        

                    if n_chunk_cache is None:
                        qp_chunk_embeds = torch.concat([q_chunk_embeds, p_chunk_embeds], dim=1)
                        qp_chunk_cache = torch.concat([q_chunk_cache, p_chunk_cache], dim=1)
                        surrogate = torch.dot(qp_chunk_embeds.flatten(), qp_chunk_cache.flatten())
                    else:
                        qpn_chunk_embeds = torch.concat([q_chunk_embeds.unsqueeze(1), p_chunk_embeds.unsqueeze(1), n_chunk_embeds], dim=1)
                        qpn_chunk_cache = torch.concat([q_chunk_cache.unsqueeze(1), p_chunk_cache.unsqueeze(1), n_chunk_cache], dim=1)
                        surrogate = torch.dot(qpn_chunk_embeds.flatten(), qpn_chunk_cache.flatten())

                    accelerator.backward(surrogate)

            # TODO: NO Effection
            if args.clip_gradient:
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                
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