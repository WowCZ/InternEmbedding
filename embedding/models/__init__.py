from embedding.models.modeling_bert import BertEmbedder
from embedding.models.modeling_mistral import MistralEmbedder
from embedding.models.modeling_bge import BGECustomEmbedder

MODEL_MAPPING = dict(
    BERT=BertEmbedder,
    Mistral=MistralEmbedder,
    BGE=BGECustomEmbedder
)

# __all__ = ['BertEmbedder', 'MistralEmbedder', 'BGECustomEmbedder']