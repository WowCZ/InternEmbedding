import os
import json
import torch
from transformers import AutoTokenizer
from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.eval.mteb_eval_wrapper import MTEBEvaluationWrapper
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# backbone = "google/bert_uncased_L-8_H-512_A-8"
# tokenizer = AutoTokenizer.from_pretrained(backbone)
# embedder_pt_file = 'ckpts/bert/embedder_11622.pt'
# device = 'cuda:0'
# embedder = BertEmbedder(backbone, pool_type='mean', checkpoint_batch_size=64, embed_dim=-1, device=device)

# mteb_evaluator = MTEBEvaluationWrapper(embedder, tokenizer=tokenizer, max_length=512, device=device, model_name='embedder_bert_uncase')
# results = mteb_evaluator.evaluation(["AskUbuntuDupQuestions", "Banking77Classification"])
# print(json.dumps(results, indent=4))

# eval_batch_size = 16
# mytryoshka_size = 4096
# mytryoshka_indexes = list(range(mytryoshka_size))
# backbone = "/fs-computility/llm/chenzhi/huggingface_cache/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e"
# tokenizer = AutoTokenizer.from_pretrained(backbone)
# tokenizer.pad_token = tokenizer.eos_token
# device = 'cuda'
# embedder = MistralEmbedder(backbone, pool_type='position_weight', checkpoint_batch_size=32, embed_dim=-1, lora_config=True, which_layer=-1, mytryoshka_indexes=mytryoshka_indexes).to(device)
# embedder.load_state_dict(torch.load('/fs-computility/llm/chenzhi/ckpts/mistral_embedder18_filter_pow1_prompt_ins_matryoshka9_temp2_lr14_ml512_2000.pt'))

# mteb_evaluator = MTEBEvaluationWrapper(embedder, tokenizer=tokenizer, max_length=512, model_name='mistral_embedder18_filter_pow1_prompt_ins_matryoshka9_temp2_lr14_ml512_2000', prompt=True, embedding_norm=False, device=device)
# results = mteb_evaluator.evaluation(["Banking77Classification", "AskUbuntuDupQuestions"])
# print(json.dumps(results, indent=4))

def evaluate_embedder(args):
    mytryoshka_indexes = list(range(args.mytryoshka_size))
    tokenizer = AutoTokenizer.from_pretrained(args.init_backbone)
    if args.backbone_type == 'BERT':
        embedder = BertEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=64, embed_dim=-1, device=args.device)
    elif args.backbone_type == 'Mistral':
        tokenizer.pad_token = tokenizer.eos_token
        embedder = MistralEmbedder(args.init_backbone, pool_type=args.pool_type, checkpoint_batch_size=32, embed_dim=-1, lora_config=args.peft_lora, which_layer=args.which_layer, mytryoshka_indexes=mytryoshka_indexes).to(args.device)
    
    if os.path.exists(args.embedder_ckpt_path):
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    mteb_evaluator = MTEBEvaluationWrapper(embedder, tokenizer=tokenizer, max_length=args.max_length, model_name=args.embedder_name, prompt=args.task_prompt, embedding_norm=args.embedding_norm, device=args.device)
    results = mteb_evaluator.evaluation(args.mteb_evaluation_tasks)
    print(json.dumps(results, indent=4))