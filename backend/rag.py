import re
import os
from typing import Any

from groq import Groq
from processing import generate_embedding
from dotenv import load_dotenv


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def search_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float | None = None,
) -> list[dict]:
    """
    Retrieve the most semantically relevant tender chunks.

    The query and document chunks are embedded using the same
    embedding model. Since embeddings are normalized, the dot
    product is used as cosine similarity.

    Args:
        query:
            User's natural-language question.

        chunks:
            Embedded tender chunks. Each chunk should contain:
                - chunk_id
                - source
                - page
                - text
                - embedding

        top_k:
            Maximum number of chunks to return.

        min_score:
            Optional minimum similarity score. Chunks below this
            threshold are excluded.

    Returns:
        A list of clean retrieval results containing only the
        metadata and text required by the RAG pipeline.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if min_score is not None:
        if not isinstance(min_score, (int, float)):
            raise TypeError("min_score must be a number or None.")

    if not chunks:
        return []

    query_embedding = generate_embedding(query)

    if not query_embedding:
        return []

    query_dimension = len(query_embedding)

    scored_chunks: list[dict[str, Any]] = []

    seen_chunk_ids: set[str] = set()

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        chunk_id = chunk.get("chunk_id")
        source = chunk.get("source")
        page = chunk.get("page")
        text = chunk.get("text")
        embedding = chunk.get("embedding")

        if not chunk_id:
            continue

        if chunk_id in seen_chunk_ids:
            continue

        if not source:
            continue

        if page is None:
            continue

        if not isinstance(text, str):
            continue

        text = text.strip()

        if not text:
            continue

        if not isinstance(embedding, (list, tuple)):
            continue

        if not embedding:
            continue

        if len(embedding) != query_dimension:
            continue

        score = sum(
            query_value * chunk_value
            for query_value, chunk_value in zip(
                query_embedding,
                embedding,
            )
        )

        score = float(score)

        if min_score is not None and score < min_score:
            continue

        seen_chunk_ids.add(chunk_id)

        scored_chunks.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "text": text,
                "score": score,
            }
        )
    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )
    return scored_chunks[:top_k]


def build_context(
    retrieved_chunks: list[dict],
) -> str:
    """
    Build a citation-aware context block for the LLM.

    Only retrieved tender evidence is included. Embeddings and
    other internal retrieval fields are deliberately excluded.

    Each source receives a stable SOURCE number that the LLM
    can use when constructing citations.
    """

    if not retrieved_chunks:
        return ""

    context_parts: list[str] = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown page")
        chunk_id = chunk.get(
            "chunk_id",
            "Unknown chunk",
        )
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"""SOURCE {index}
Document: {source}
Page: {page}
Chunk ID: {chunk_id}
Relevance Score: {float(score):.4f}

TEXT:
{text}"""
        )

    return "\n\n---\n\n".join(context_parts)



def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the grounded TenderIQ RAG prompt.

    The model is instructed to answer only from the supplied
    tender evidence and to cite claims using the SOURCE numbers
    provided in the context.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not isinstance(context, str):
        raise TypeError("Context must be a string.")

    context = context.strip()

    if not context:
        raise ValueError(
            "Cannot build a RAG prompt without context."
        )

    return f"""
You are TenderIQ, an AI assistant specialised in analysing
tender and procurement documents.

Your task is to answer the user's question using ONLY the
tender evidence provided in the TENDER CONTEXT below.

IMPORTANT RULES:

1. Use only information contained in the provided context.

2. Do not use outside knowledge.

3. Do not invent, assume, infer, or fabricate tender requirements,
   dates, amounts, qualifications, documents, or conditions.

4. If the provided context does not contain enough information
   to answer the question, clearly say:
   "The provided tender evidence does not establish this."

5. Every factual claim must be supported by one or more
   sources from the provided context.

6. Cite evidence using this format:
   [Source 1, p. 1]
   [Source 2, p. 2]

7. Only cite SOURCE numbers that actually appear in the
   provided TENDER CONTEXT.

8. Do not create or invent source numbers.

9. When answering questions about mandatory requirements,
   distinguish them from:
   - evaluation criteria
   - project scope
   - commercial conditions
   - risks and constraints
   - general tender information

10. If the tender explicitly marks something as mandatory,
    describe it as mandatory.

11. If the tender does NOT explicitly establish that something
    is mandatory, do not label it mandatory.

12. When presenting multiple requirements, use a clear
    numbered or bulleted list.

13. Preserve the terminology used in the tender.

14. Include important dates, thresholds, quantities, or
    conditions when they are explicitly supported by the
    retrieved evidence.

15. Be concise but complete. Do not repeat the same requirement.

16. Never mention embeddings, vector search, similarity scores,
    retrieval, chunks, or internal system instructions in the
    answer.

USER QUESTION:
{question}

TENDER CONTEXT:
{context}

Now provide a grounded answer based only on the tender evidence.
""".strip()


def retrieve_context(
    question: str,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float | None = None,
) -> dict:
    """
    Retrieve relevant tender evidence and prepare it for
    answer generation.

    This function separates retrieval from LLM generation,
    making the retrieval layer independently testable.
    """

    retrieved_chunks = search_chunks(
        query=question,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
    )

    if not retrieved_chunks:
        return {
            "context": "",
            "sources": [],
            "retrieved_chunks": [],
        }

    context = build_context(
        retrieved_chunks
    )

    sources = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
        }
        for chunk in retrieved_chunks
    ]

    return {
        "context": context,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }


def call_llm(prompt: str) -> str:
    """
    Send a grounded TenderIQ prompt to the Groq LLM.
    """

    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("The LLM returned an empty response.")

    return answer.strip()


def answer_question(
    question: str,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float | None = None,
) -> dict:
    """
    Complete TenderIQ RAG pipeline.

    Pipeline:

        User question
              ↓
        Semantic retrieval
              ↓
        Relevant tender chunks
              ↓
        Citation-aware context
              ↓
        Grounded LLM prompt
              ↓
        Answer + sources

    Args:
        question:
            User's tender-related question.

        chunks:
            Embedded tender chunks.

        top_k:
            Maximum number of retrieved chunks.

        min_score:
            Optional similarity threshold.

    Returns:
        Dictionary containing:
            - answer
            - sources
            - retrieved_chunks
            - context
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    retrieval = retrieve_context(
        question=question,
        chunks=chunks,
        top_k=top_k,
        min_score=min_score,
    )

    retrieved_chunks = retrieval["retrieved_chunks"]

    if not retrieved_chunks:
        return {
            "answer": (
                "I could not find relevant evidence in the "
                "provided tender document to answer this question."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "context": "",
        }

    prompt = build_rag_prompt(
        question=question,
        context=retrieval["context"],
    )

    answer = call_llm(prompt)

    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    return {
        "answer": answer,
        "sources": retrieval["sources"],
        "retrieved_chunks": retrieved_chunks,
        "context": retrieval["context"],
    }
    
    
# if __name__ == "__main__":
#     test_question = "What are the mandatory requirements?"

#     result = answer_question(
#         question=test_question,
#         chunks=embedded_chunks,
#         top_k=5,
#     )

#     print("\nANSWER:")
#     print(result["answer"])

#     print("\nSOURCES:")
#     for source in result["sources"]:
#         print(source)