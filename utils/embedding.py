import ollama
import numpy as np
from typing import List
from config import EMBEDDING_MODEL
from utils.logger_config import logger


def embed_chunks(text_list: List[str]) -> List[np.ndarray]:
    """
    Generates embeddings for a batch of text chunks using Ollama's model.

    Args:
        text_list (List[str]): List of text strings.

    Returns:
        List[np.ndarray]: List of NumPy arrays representing embeddings.
    """
    embeddings = []

    for text in text_list:
        try:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            if "embedding" in response:
                embeddings.append(np.array(response["embedding"]))
            else:
                raise ValueError(f"Missing 'embedding' key in response: {response}")
        except Exception as e:
            logger.error(f"Error generating embedding for text: {text[:50]}... - {e}")
            embeddings.append(np.zeros(EMBEDDING_MODEL.dimension))

    return embeddings
