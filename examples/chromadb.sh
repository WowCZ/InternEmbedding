# !/bin/sh

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=5 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
CUDA_VISIBLE_DEVICES=7 /root/miniconda3/envs/embedding/bin/python /fs-computility/llm/chenzhi/InternEmbedding/app.py &
wait