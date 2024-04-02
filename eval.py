import os
import json
import torch
from embedding.train.training_embedder import initial_model
from embedding.eval.mteb_eval_wrapper import PrefEvaluationWrapper
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def evaluate_embedder(args):
    '''
    # !Deprecated
    mytryoshka_indexes = list(range(args.mytryoshka_size))
    tokenizer = AutoTokenizer.from_pretrained(args.init_backbone)
    if args.backbone_type == 'BERT':
        embedder = BertEmbedder(args.init_backbone, 
                                pool_type=args.pool_type, 
                                checkpoint_batch_size=10, 
                                embed_dim=-1, 
                                lora_config=args.peft_lora, 
                                which_layer=args.which_layer, 
                                mytryoshka_indexes=mytryoshka_indexes,
                                normalize=args.embedding_norm)
    elif args.backbone_type == 'Mistral':
        tokenizer.pad_token = tokenizer.eos_token
        embedder = MistralEmbedder(args.init_backbone, 
                                   pool_type=args.pool_type, 
                                   checkpoint_batch_size=10, 
                                   embed_dim=-1, 
                                   lora_config=args.peft_lora, 
                                   which_layer=args.which_layer, 
                                   mytryoshka_indexes=mytryoshka_indexes,
                                   normalize=args.embedding_norm)
    elif args.backbone_type == 'BGE':
        embedder = BGECustomEmbedder(args.init_backbone, 
                                     pool_type=args.pool_type, 
                                     checkpoint_batch_size=10, 
                                     embed_dim=-1, 
                                     lora_config=args.peft_lora, 
                                     which_layer=args.which_layer, 
                                     mytryoshka_indexes=mytryoshka_indexes,
                                     normalize=args.embedding_norm)
    '''
    embedder, tokenizer = initial_model(args)

    embedder = embedder.to(args.device)
    embedder.eval()
    if args.embedder_ckpt_path and os.path.exists(args.embedder_ckpt_path):
        print(f'Loading embedder checkpoint from {args.embedder_ckpt_path}')
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    mteb_evaluator = PrefEvaluationWrapper(embedder, tokenizer=tokenizer, max_length=args.max_length, model_name=args.embedder_name, prompt=args.task_prompt, device=args.device, result_dir=args.result_dir, params=args)
    results = mteb_evaluator.evaluation()
    print(json.dumps(results, indent=4))
    os.makedirs(args.result_dir, exist_ok=True)
    
    result_file = os.path.join(args.result_dir, f'{args.embedder_name}_results.jsonl')
    with open(result_file, 'a') as f:
        results['ckpt'] = args.embedder_ckpt_path
        f.write(json.dumps(results) + '\n')