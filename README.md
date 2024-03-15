<h1 align="center"> <img src="./resets/images/embedder_triangle2.png" alt="embedder" width="50"> InternEmbedding </h1>

## 🚀 Embedder Training & Evaluation
-------
### Train the embedder:
```shell
accelerate launch --config_file /path_to/accelerate_config.yaml run.py train \
--init_backbone=BAAI/bge-base-zh-v1.5 \
--pool_type=cls \
--backbone_type=BGE \
--checkpoint_batch_size=128 \
--gradcache_chunk_size=-1 \
--temperature=0.01 \
--learning_rate=1e-5 \
--hard_negative_sampling \
--hard_negative_num=2 \
--matryoshka_adaptive_dims=768 \
--mytryoshka_size=768 \
--batch_size_per_gpu=512 \
--embedder_name=your_logging_name
```

### Evaluate the embedder on MTEB:
```shell
python run.py evaluate \
--pool_type=eos \
--embedder_ckpt_path=/path_to/your_saved_ckpt \
--embedder_name=your_logging_name \
--mteb_evaluation_tasks=Banking77Classification,EmotionClassification \
```