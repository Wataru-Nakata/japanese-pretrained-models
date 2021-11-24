
from typing import List
from sentence_transformers import SentenceTransformer
import torch
class LabseExtractor():
    ''' Copied from https://tfhub.dev/google/LaBSE/2'''
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
    def extract_embeds(self,sentences:List[str]):
        with torch.cuda.amp.autocast():
            output = self.model.encode(sentences,batch_size=512,normalize_embeddings=True)
        return output

