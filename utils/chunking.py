import nltk
from typing import List

nltk.download("punkt")
def sliding_window_chunking(text: str, CHUNK_SIZE, CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into overlapping chunks using a sliding window approach.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk in characters.
        overlap (int): The overlap between consecutive chunks in characters.

    Returns:
        List[str]: A list of chunked text segments.
    """
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > CHUNK_SIZE:
            # Save the current chunk and start a new one
            chunks.append(" ".join(current_chunk))

            # Keep the last `overlap` characters for continuity
            overlap_text = (
                "".join(current_chunk)[-CHUNK_OVERLAP:] if current_chunk else ""
            )
            current_chunk = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)

        # Add the sentence to the chunk
        current_chunk.append(sentence)
        current_length += sentence_length

    # Append the last chunk if there's remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
