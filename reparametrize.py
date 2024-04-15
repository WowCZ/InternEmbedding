import os
import json
import torch
from embedding.train.training_embedder import initial_model
from embedding.eval.mteb_eval_wrapper import PrefEvaluationWrapper
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoModel
import copy

def reparametrize(state_dict):
    pass


def test_output_pref(model):
    weight = model.encoder.backbone.model.model.layers[0].attention.wqkv.base_layer
    print(weight.weight.data)
    print(f"STD: {float(weight.weight.data.std())} | MEAN: {float(weight.weight.data.mean())}")

def test_output_pref_1(model):
    weight = model.encoder.backbone.model.layers[0].attention.wqkv
    print(weight.weight.data)
    print(f"STD: {float(weight.weight.data.std())} | MEAN: {float(weight.weight.data.mean())}")

def test_output_hf(model):
    weight = model.model.layers[0].attention.wqkv
    print(weight.weight.data)
    print(f"STD: {float(weight.weight.data.std())} | MEAN: {float(weight.weight.data.mean())}")

def reparametrize_func(args):
    # ------ innitialize the model ------ #
    embedder, tokenizer = initial_model(args)
    # embedder = embedder.to(args.device)
    embedder.eval()
    if args.embedder_ckpt_path and os.path.exists(args.embedder_ckpt_path):
        print(f'Loading embedder checkpoint from {args.embedder_ckpt_path}')
        embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

    train_embedder = copy.deepcopy(embedder)
    train_embedder.train()

    # ----- merge and unload peft model ----- #
    train_embedder.encoder.backbone = train_embedder.encoder.backbone.merge_and_unload()

    untrained_hf_internlm2_prefmodel_pth = '/fs-computility/llm/shared/wangyikun/ckpts/internlm2-preference-1_8b-sft-untrained'
    target_hf_internlm2_prefmodel_pth = '/fs-computility/llm/shared/wangyikun/ckpts/internlm2-preference-V1_0-1_8b-reparametrized'
    hf_pref_model = AutoModel.from_pretrained(untrained_hf_internlm2_prefmodel_pth, trust_remote_code=True)
    print(train_embedder)
    print(hf_pref_model)

    test_output_pref_1(train_embedder)
    test_output_hf(hf_pref_model)

    # ------- reparametrization ------- #
    embdder_state_dict = train_embedder.state_dict()
    pref_model_state_dict = {}
    for k, v in embdder_state_dict.items():
        if k == 'project.weight':
            pref_model_state_dict['score.weight'] = v
        else:
            assert k.startswith('encoder.backbone.')
            k = k.replace('encoder.backbone.', '')
            pref_model_state_dict[k] = v
    
    hf_pref_model.load_state_dict(pref_model_state_dict, strict=False)
    # -------- single test ---------- #

    test_inputs = ["Hello, My name is Aaron. I am a student at the University of Illinois at Urbana-Champaign.",
                   "MIT is a great university. It has the best computer science program in the world.",
                   "P&G has been a great company to work for. One of the product that customers likes so much is the Tide detergent. Just buy it now! 80% Discount!",]
    
    for input in test_inputs:
        tokens = tokenizer(input, return_tensors='pt')

        embedder_logits, _, _ = train_embedder(tokens, None, None)
        hf_logits = hf_pref_model(tokens['input_ids']).logits
        print(f"Embedder Preference V1.0 logits: {embedder_logits}")
        print(f"Huggingface Preference V1.0 logits: {hf_logits}")
        print("---------------- Test Pass !!! --------------")
        print()
        # print(embedder_logits)
        # print(hf_logits)

    hf_pref_model.save_pretrained(target_hf_internlm2_prefmodel_pth)
    tokenizer.save_pretrained(target_hf_internlm2_prefmodel_pth)
