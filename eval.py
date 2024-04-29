import os
import json
import torch
from embedding.train.training_embedder import initial_model
from embedding.eval.mteb_eval_wrapper import MTEBEvaluationWrapper
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def evaluate_embedder(args):
    embedder, tokenizer = initial_model(args)
    embedder = embedder.to(args.device)

    if args.embedder_ckpt_path and os.path.exists(args.embedder_ckpt_path):
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    embedder.eval()

    mteb_evaluator = MTEBEvaluationWrapper(embedder, tokenizer=tokenizer, max_length=args.max_length, model_name=args.embedder_name, prompt=args.task_prompt, device=args.device, result_dir=args.result_dir)
    results = mteb_evaluator.evaluation(args.mteb_evaluation_tasks)
    print(json.dumps(results, indent=4))