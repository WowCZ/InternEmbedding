import math
import json
import tqdm
import torch
import chromadb
from embedding.models.modeling_bge import BGEEmbedder
from chromadb import Documents, EmbeddingFunction, Embeddings
from apps.vector_db.text_loading_chroma import BGEFunction


def test_math():
    chroma_path = f'/fs-computility/llm/shared/chenzhi/chromadb'
    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_collection(name="gaokao", embedding_function=BGEFunction(bge_name='BAAI/bge-base-zh-v1.5'))

    # 在区间（0,1）与（1,2）个随机取一个数，则两数只和大于7/4的概率为 \nA. 7/9 \nB. 23/32 \nC. 9/32 \nD. 2/9
    # 设函数f(x) = (1-x) / (1+x)，则下列函数中为奇函数的是 \nA. f(x-1)-1 \nB. f(x-1)+1 \nC. f(x+1)-1 \nD. f(x+1)+1
    # 设a!=0，若x=a为函数f(x)=a(x-a)**2(x-b)的极大值点，则（）\nA. a<b \nB. a>b \nC. ab<a**2 \nD. ab>a**2
    query_texts = collection.query(
            query_texts=["帮我收集与极大值知识点相关的题目"],
            n_results=10
        )

    print(json.dumps(query_texts, indent=4, ensure_ascii=False))