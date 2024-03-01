import os
import json
import torch
from transformers import AutoTokenizer
from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.eval.mteb_eval_wrapper import MTEBEvaluationWrapper
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def evaluate_embedder(args):
    mytryoshka_indexes = list(range(args.mytryoshka_size))
    tokenizer = AutoTokenizer.from_pretrained(args.init_backbone)
    if args.backbone_type == 'BERT':
        embedder = BertEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=64, embed_dim=-1, device=args.device)
    elif args.backbone_type == 'Mistral':
        tokenizer.pad_token = tokenizer.eos_token
        embedder = MistralEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=-1, embed_dim=-1, lora_config=args.peft_lora, which_layer=args.which_layer, mytryoshka_indexes=mytryoshka_indexes).to(args.device)
    
    if os.path.exists(args.embedder_ckpt_path):
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    mteb_evaluator = MTEBEvaluationWrapper(embedder, tokenizer=tokenizer, max_length=args.max_length, model_name=args.embedder_name, prompt=args.task_prompt, embedding_norm=args.embedding_norm, device=args.device)
    results = mteb_evaluator.evaluation(args.mteb_evaluation_tasks)
    print(json.dumps(results, indent=4))