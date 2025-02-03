import os

# Chunking Parameters
CHUNK_SIZE = 512  # Max tokens per chunk
CHUNK_OVERLAP = 128  # Overlapping tokens for sliding window

# Embedding Model
EMBEDDING_MODEL = "nomic-embed-text:latest"

AI_MODEL = "qwen2.5:latest"
# AI_MODEL = 'deepseek-r1:7b'
# AI_MODEL="deepseek-r1:14b"

# Database Path
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "rag_pdfs"


# Number of retrieval results
TOP_K = 5

PDF_FOLDER = "data/pdfs"

# Logging Settings
LOGGING_DIR = "logs"
if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR)
