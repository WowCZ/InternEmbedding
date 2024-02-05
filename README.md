<h1 align="center"> InternEmbedding </h1>

## 🚀Embedder Training & Evaluation
-------
### Training the embedder:
```shell
python run.py train \
    --learning_rate=2e-4 \
    --temperature=0.01
```

### Evaluate the embedder on MTEB:
```shell
python run.py evaluation \
    --max_lengthe=1024
```