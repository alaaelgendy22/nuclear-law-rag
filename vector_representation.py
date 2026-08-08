import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import config

def load_embedding_model(model_name: str = config.EMBEDDING_MODEL):
    return SentenceTransformer(model_name)

def compute_embeddings(chunks: List[Dict[str, Any]], model: SentenceTransformer = None) -> np.ndarray:
    if model is None:
        model = load_embedding_model()
    texts = [chunk['search_text'] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings
