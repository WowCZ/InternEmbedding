import torch
from transformers import AutoModel, AutoTokenizer

class BGEEmbedder():
    def __init__(self, backbone: str, device: str, max_length: int) -> None:
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = AutoModel.from_pretrained(backbone)
        model.eval()
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, sentences):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
        # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
        for k in encoded_input.keys():
            encoded_input[k] = encoded_input[k].to(self.model.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.detach().cpu()