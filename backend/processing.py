import re

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def create_chunks(
    pages: list[dict],
    source_filename: str,
    chunk_size: int = 1200,
) -> list[dict]:
    """
    Split cleaned tender text into page-aware chunks.

    Chunks do not overlap. Each chunk preserves its source
    filename and page number so retrieved information can
    later be cited back to the tender.
    """

    chunks = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"].strip()

        if not page_text:
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", page_text)
            if paragraph.strip()
        ]

        current_chunk = ""
        chunk_number = 1

        for paragraph in paragraphs:

            if not current_chunk:
                current_chunk = paragraph
                continue

            proposed_chunk = f"{current_chunk}\n\n{paragraph}"

            if len(proposed_chunk) <= chunk_size:
                current_chunk = proposed_chunk
                continue

            chunks.append(
                {
                    "chunk_id": f"page-{page_number}-chunk-{chunk_number}",
                    "source": source_filename,
                    "page": page_number,
                    "text": current_chunk,
                }
            )

            chunk_number += 1
            current_chunk = paragraph

        # Save the final chunk from this page.
        if current_chunk:
            chunks.append(
                {
                    "chunk_id": f"page-{page_number}-chunk-{chunk_number}",
                    "source": source_filename,
                    "page": page_number,
                    "text": current_chunk,
                }
            )

    return chunks



def generate_embedding(text: str) -> list[float]:
    """
    Generate a normalized semantic embedding for a text string.

    Uses the shared SentenceTransformer model so document chunks
    and user queries are embedded in the same vector space.
    """

    if not isinstance(text, str):
        raise TypeError("Embedding input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Cannot generate an embedding for empty text.")

    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embedding.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for all tender chunks.

    Each chunk keeps its original metadata and receives
    a normalized semantic embedding.
    """

    if not chunks:
        return []

    embedded_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        embedding = generate_embedding(text)

        embedded_chunks.append(
            {
                **chunk,
                "embedding": embedding,
            }
        )

    return embedded_chunks