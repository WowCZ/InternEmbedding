# !/bin/sh
source activate /root/miniconda3/envs/embedding

export http_proxy=100.66.27.20:3128
export https_proxy=100.66.27.20:3128

python /fs-computility/llm/chenzhi/InternEmbedding/app.py