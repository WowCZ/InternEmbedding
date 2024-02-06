<h1 align="center"> <img src="./resets/images/embedder_triangle2.png" alt="embedder" width="50"> InternEmbedding </h1>

## 🚀Embedder Training & Evaluation
-------
### Train the embedder:
```shell
python run.py train \
    --learning_rate=2e-4 \
    --temperature=0.01
```

### Evaluate the embedder on MTEB:
```shell
python run.py evaluation \
    --embedder_ckpt_path=xxx \
    --embedder_name=xxx \
    --max_length=512
```