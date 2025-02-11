# RAG-based Document Retrieval System

This project implements a Retrieval-Augmented Generation (RAG) system for querying documents. 
It combines semantic search (embeddings) and BM25 keyword retrieval to retrieve the most relevant legal text and generate AI-driven responses.

## 🚀 Features

- Hybrid Retrieval (Embeddings + BM25) – Improves accuracy by combining vector-based retrieval with keyword matching.
- Cross-Encoder Re-ranking – Ranks retrieved documents for better relevance.
- Document Chunking with Sliding Window – Ensures context-aware retrieval.
- Efficient Embedding Generation – Uses Ollama’s embedding model for high-performance vector search.
- ChromaDB as Vector Store – Stores document embeddings for fast retrieval.
- BM25-based Keyword Search – Supports keyword-based retrieval when embeddings are insufficient.
- Robust Logging & Error Handling – Improves debugging and system monitoring.
- Unit Tests with Pytest – Ensures functionality is reliable.

## 🛠️ Installation

1. Clone the Repository
```bash
git clone https://github.com/ahmedashraffcih/rag-document-assistant.git
cd rag-document-assistant
```
2. Create a Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# For Windows: venv\Scripts\activate

pip install -r requirements.txt
```

3. Install Additional Dependencies (if needed)
```bash
python -m spacy download en_core_web_sm  # If using spaCy for text processing
```

## 📂 Project Structure

```bash
📦 rag-document-assistant
├── main.py                 # Entry point for querying the system
├── rag_pipeline.py         # Main pipeline for processing PDFs and querying
├── config.py               # Configuration settings
├── data/
│   ├── pdfs/               # Folder containing PDFs
│   ├── embeddings/         # (Optional) Stores precomputed embeddings
├── utils/
│   ├── chunking.py         # Splits documents into chunks
│   ├── embeddings.py       # Generates text embeddings
│   ├── retrieval.py        # Retrieves relevant text
│   ├── logger_config.py    # Logging setup
│   ├── text_processing.py  # (Optional) Preprocessing functions
├── logs/                   # Log files
├── tests/                  # Unit tests for different components
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 📌 Configuration (config.py)
Modify config.py to adjust system parameters:

```bash
# AI Model for response generation
AI_MODEL = "qwen2.5:latest"

# Embedding Model for retrieval
EMBEDDING_MODEL = "nomic-embed-text:latest"

# ChromaDB Storage Path
CHROMA_DB_PATH = "data/chromadb"

# Chunking Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
```

## 📑 How to Process Documents

1. Add PDF Documents
Place PDFs in the data/pdfs/ directory.

2. Run PDF Processing
```bash
python main.py --process-pdfs
```
This will:

- Extract text from PDFs
- Chunk the text into smaller segments
- Generate embeddings for each chunk
- Store embeddings in ChromaDB for fast retrieval
- Index chunks using BM25 for keyword search


## 📝 Querying the RAG System

To ask questions:

```bash 
python main.py 
```
Then enter your question when prompted:

```bash 
🔹 Enter your query: Who should install the CCTV Surveillance Camera and devices?
```

### Sample AI Response
```bash
Document: Law No. 9 of 2011 regulating the use of Security and Surveillance CCTV Camera and devices
Article: 3

The law states that Facilities owners and those in charge of management are responsible for installing CCTV Surveillance Cameras and related devices.
```

## Chunking Technique
The system uses a sliding window chunking approach to divide large text documents into smaller, overlapping chunks. 
This ensures better context preservation across chunks and improves retrieval relevance.

How It Works:

- Sentence Tokenization: The text is first split into sentences using spaCy or NLTK.
- Sliding Window: A fixed chunk size is used to create segments, with an overlap between consecutive chunks.
- Overlap for Context: The overlap ensures that key information does not get lost between chunks.

## Retrieval and Search Process
Your system retrieves relevant chunks using a hybrid retrieval approach, combining vector embeddings (semantic search) and BM25 (keyword search).

Retrieval Steps:

1. Semantic Search (ChromaDB)
    - Uses Ollama embeddings to represent text in a high-dimensional space.
    - Computes cosine similarity between the query embedding and stored chunk embeddings.
    - Returns the top k most relevant chunks.
2. Keyword Search (BM25)
    - Tokenizes stored chunks and indexes them using BM25, a traditional term-based retrieval method.
    - Scores documents based on keyword matching.
    - Returns the top k chunks based on term frequency and relevance.
3. Merging Results
    - The system combines results from semantic search (ChromaDB) and BM25 using a ranking function.
    - The higher score from either method is used to prioritize results.

## Reranking Mechanism
Once retrieved chunks are obtained, they are reranked using a Cross-Encoder model. 
This improves the ranking by assessing contextual relevance between the query and each chunk.

Steps:

- Query-Chunk Pairs: The query is paired with each retrieved chunk.
- Cross-Encoder Model: A Transformer-based model (like cross-encoder/ms-marco-MiniLM-L-6-v2) evaluates the query-chunk relevance.
- New Ranking Scores: The model assigns a new similarity score to each chunk.
- Sorting: The chunks are sorted based on the new scores, ensuring the most relevant ones appear first.

## 🧪 Running Tests

To ensure everything is working, run:
```bash
pytest
```
### Test Coverage
- Chunking (test_chunking.py) – Ensures text is split correctly.
- Embeddings (test_embeddings.py) – Validates vector representations.
- Retrieval (test_retrieval.py) – Tests search accuracy.
- Pipeline (test_rag_pipeline.py) – End-to-end testing.

## 🤝 Contributing

Want to improve this system?

1. Fork the repository
2. Create a feature branch (git checkout -b feature-name)
3. Commit changes (git commit -m "Add feature XYZ")
4. Push to the branch (git push origin feature-name)
5. Open a Pull Request (PR)
