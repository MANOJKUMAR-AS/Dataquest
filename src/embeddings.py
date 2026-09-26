import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings

    @staticmethod
    def cosine_similarity(a, b):
        # Because embeddings are normalized,
        # cosine similarity is simply the dot product.
        return float(np.dot(a, b))