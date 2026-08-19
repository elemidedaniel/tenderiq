from pathlib import Path
import re
from typing import Any
from sentence_transformers import SentenceTransformer
import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile


app = FastAPI(
    title="TenderIQ",
    description="AI-powered construction tender intelligence platform",
    version="0.1.0",
)


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

tender_vector_store: list[dict] = []


def extract_pdf_text(file_bytes: bytes) -> dict:
    """
    Extract text from every page of a PDF.

    Page-level text is preserved because TenderIQ will later
    use page references for RAG citations and evidence.
    """

    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            "The uploaded file is not a valid PDF."
        ) from exc

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            raw_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            cleaned_text = clean_text(raw_text)

            pages.append(
                {
                    "page": page_number,
                    "text": cleaned_text,
                }
            )

    finally:
        document.close()

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    return {
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
    }


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from a PDF.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # Remove excessive spaces while preserving newlines.
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



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



def find_value(text: str, pattern: str) -> str | None:
    """
    Extract a single value from a labeled field.

    Example:
    'Tender Reference: TND-2026-014'
    """

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not match:
        return None

    value = match.group(1).strip()

    return value if value else None




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
    
    
def test_retrieval(
    chunks: list[dict],
    questions: list[str],
    top_k: int = 5,
) -> None:
    """
    Test TenderIQ retrieval without calling the LLM.

    Prints the retrieved chunks, relevance scores,
    pages, and source text for each question.
    """

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = search_chunks(
            query=question,
            chunks=chunks,
            top_k=top_k,
        )

        if not results:
            print("NO RELEVANT CHUNKS FOUND.")
            continue

        for index, result in enumerate(results, start=1):
            print(f"\n[{index}]")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Chunk: {result['chunk_id']}")
            print("-" * 80)
            print(result["text"])
            
test_questions = [
    "What are the mandatory requirements?",
    "What documents must be submitted?",
    "What is the bid submission deadline?",
    "What are the key risks and constraints?",
    "What are the commercial conditions?",
]

test_retrieval(
    chunks=embedded_chunks,
    questions=test_questions,
    top_k=5,
)





def extract_metadata(pages: list[dict]) -> dict:
    """
    Extract deterministic metadata from the tender.

    Handles common variations in tender terminology while
    keeping extraction rule-based and predictable.
    """

    full_text = "\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    # Project name
    project_name = find_value(
        full_text,
        r"Project\s+Name\s*:\s*(.+)",
    )

    # Some tenders use the document title instead of
    # an explicit "Project Name:" field.
    if not project_name:
        title_match = re.search(
            r"Invitation\s+to\s+Tender\s*\n+(.+?)(?:\n|$)",
            full_text,
            flags=re.IGNORECASE,
        )

        if title_match:
            project_name = title_match.group(1).strip()

    # Some documents use "REQUEST FOR PROPOSAL"
    # followed immediately by the project title.
    if not project_name:
        title_match = re.search(
            r"Request\s+for\s+Proposal\s*\n+(.+?)(?:\n|$)",
            full_text,
            flags=re.IGNORECASE,
        )

        if title_match:
            project_name = title_match.group(1).strip()

    # Tender reference
    tender_reference = find_value(
        full_text,
        r"(?:Tender\s+Reference|Tender\s+No\.?|Tender\s+Number|RFP\s+No\.?)\s*:\s*(.+)",
    )

    # Issuing organization / client
    issuing_organization = find_value(
        full_text,
        r"(?:Issuing\s+Organization|Client|Employer|Procuring\s+Entity)\s*:\s*(.+)",
    )

    # Project location
    location = find_value(
        full_text,
        r"(?:Project\s+Location|Project\s+Site|Location|Site)\s*:\s*(.+)",
    )

    # Submission deadline
    submission_deadline = find_value(
        full_text,
        r"(?:Submission\s+Deadline|Bid\s+Submission\s+Deadline|"
        r"Tender\s+Submission\s+Deadline)\s*:\s*(.+)",
    )

    # Contract value / project budget
    contract_value = find_value(
        full_text,
        r"(?:Estimated\s+Contract\s+Value|Estimated\s+Project\s+Budget|"
        r"Contract\s+Value|Project\s+Value|Budget)\s*:\s*(.+)",
    )

    return {
        "project_name": project_name,
        "tender_reference": tender_reference,
        "issuing_organization": issuing_organization,
        "location": location,
        "submission_deadline": submission_deadline,
        "contract_value": contract_value,
    }


def extract_requirements(pages: list[dict]) -> list[dict]:
    """
    Extract mandatory tender requirements while preserving
    the page where each requirement was found.

    Supports common tender formats where requirements appear
    in a section containing a Requirement / Mandatory table.
    """

    requirements = []

    for page in pages:
        page_text = page["text"]

        # Look for the mandatory requirements section.
        section_match = re.search(
            r"(?:Mandatory\s+Requirements|Eligibility\s+and\s+Mandatory\s+Requirements)"
            r"(.*?)(?=\n\s*\d+\.\s+[A-Z][^\n]*|\Z)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not section_match:
            continue

        section_text = section_match.group(1)

        # Match requirement rows ending with "Yes".
        #
        # Example:
        # CAC registration and valid company documentation Yes
        #
        # The non-greedy match prevents multiple rows from
        # accidentally being combined.
        matches = re.finditer(
            r"([A-Za-z][^\n]+?)\s+(Yes|No)\s*$",
            section_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        for match in matches:
            requirement = match.group(1).strip()
            mandatory_value = match.group(2).strip().lower()

            # Clean excessive whitespace.
            requirement = re.sub(
                r"\s+",
                " ",
                requirement,
            )

            # Ignore table headers.
            if requirement.lower() in {
                "requirement",
                "mandatory",
            }:
                continue

            item = {
                "requirement": requirement,
                "mandatory": mandatory_value == "yes",
                "page": page["page"],
            }

            # Prevent duplicates.
            already_exists = any(
                existing["requirement"].lower() == requirement.lower()
                for existing in requirements
            )

            if not already_exists:
                requirements.append(item)

    return requirements


def extract_risks(pages: list[dict]) -> list[dict]:
    """
    Extract explicitly mentioned project risks and constraints
    while preserving their source page.

    Supports common tender sections such as:
    - Key Risks and Constraints
    - Risks and Constraints
    - Key Risks
    """

    risks = []

    for page in pages:
        page_text = page["text"]

        # Find the risks/constraints section.
        section_match = re.search(
            r"(?:Key\s+Risks\s+and\s+Constraints|"
            r"Risks\s+and\s+Constraints|"
            r"Key\s+Risks)"
            r"(.*?)(?=\n\s*\d+\.\s+[A-Z][^\n]*|\Z)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not section_match:
            continue

        section_text = section_match.group(1)

        # Remove introductory language that is not itself a risk.
        section_text = re.sub(
            r"^(?:\s*Contractors?\s+should\s+account\s+for|"
            r"\s*Bidders?\s+should\s+consider)\s+",
            "",
            section_text,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace.
        section_text = re.sub(
            r"\s+",
            " ",
            section_text,
        ).strip()

        # Common risk phrases.
        #
        # We intentionally keep this controlled rather than asking
        # an LLM to invent or infer risks during Day 1.
        risk_patterns = [
            r"material price fluctuations",
            r"inflation and fluctuations in construction material prices",
            r"weather-related delays",
            r"site access restrictions",
            r"restricted access to occupied clinical areas",
            r"procurement lead times",
            r"long lead times for imported medical equipment",
            r"coordination between multiple subcontractors",
            r"utility interruptions",
            r"coordination with hospital operations",
        ]

        for pattern in risk_patterns:
            matches = re.finditer(
                pattern,
                section_text,
                flags=re.IGNORECASE,
            )

            for match in matches:
                risk = match.group(0).strip()

                risk = re.sub(
                    r"\s+",
                    " ",
                    risk,
                )

                already_exists = any(
                    item["risk"].lower() == risk.lower()
                    for item in risks
                )

                if not already_exists:
                    risks.append(
                        {
                            "risk": risk,
                            "page": page["page"],
                        }
                    )

    return risks


def extract_tender_intelligence(pages: list[dict]) -> dict:
    """
    Combine deterministic extraction into a structured
    tender intelligence object.
    """

    metadata = extract_metadata(pages)
    requirements = extract_requirements(pages)
    risks = extract_risks(pages)

    return {
        "metadata": metadata,
        "requirements": requirements,
        "risks": risks,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "TenderIQ",
    }


@app.post("/upload")
async def upload_tender(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are currently supported.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The PDF is too large. Maximum size is 20 MB.",
        )

    try:
        # Day 1: Extract and clean the PDF
        extracted = extract_pdf_text(file_bytes)

        pages = extracted["pages"]

        # Day 1: Extract tender intelligence
        intelligence = extract_tender_intelligence(pages)

        # Day 2: Create source-aware chunks for the RAG pipeline
        chunks = create_chunks(
            pages=pages,
            source_filename=file.filename,
        )
        
        embedded_chunks = embed_chunks(chunks)

        # Day 2: Store the embedded chunks for semantic retrieval
        tender_vector_store.clear()
        tender_vector_store.extend(embedded_chunks)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": len(file_bytes),

        "page_count": extracted["page_count"],
        "text_length": len(extracted["text"]),

        "metadata": intelligence["metadata"],
        "requirements": intelligence["requirements"],
        "risks": intelligence["risks"],

        "chunk_count": len(chunks),
        "chunks": chunks,

        "pages": pages,
    }
    
@app.post("/search")
async def search_tender(query: str, top_k: int = 5):

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    if not tender_vector_store:
        raise HTTPException(
            status_code=400,
            detail="No tender has been uploaded yet.",
        )

    results = search_chunks(
        query=query,
        chunks=tender_vector_store,
        top_k=top_k,
    )

    return {
        "query": query,
        "results": results,
    }
    
pages = extract_pages(pdf_path)

chunks = create_chunks(
    pages=pages,
    source_filename="TenderIQ_Test_Tender_02.pdf",
)

embedded_chunks = embed_chunks(chunks)

print(f"Chunks: {len(chunks)}")
print(f"Embedded chunks: {len(embedded_chunks)}")