
from typing import List
from tensorflow.python.ops.math_ops import divide
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text
import numpy as np
class LabseExtractor():
    ''' Copied from https://tfhub.dev/google/LaBSE/2'''
    def __init__(self) -> None:
        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"
        )
        self.encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")
    def normalization(self,embeds):
        norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
        return embeds/norms
    def divide_to_batches(self,sentences,n=256):
        for i in range(0,len(sentences), n):
            yield sentences[i:i+ n]
    def extract_embeds(self,sentences:List[str]):
        embeds = []
        tf.config.optimizer.set_jit(True)
        for batch in self.divide_to_batches(sentences):
            embeds += self.encoder(self.preprocessor(tf.constant(batch)))["default"].numpy().tolist()
        return self.normalization(embeds)

