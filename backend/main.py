import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extraction import (
    extract_pdf_text,
    extract_tender_intelligence,
)

from processing import (
    create_chunks,
    embed_chunks,
)

from rag import (
    search_chunks,
    answer_question,
)


app = FastAPI(
    title="TenderIQ",
    description="AI-powered construction tender intelligence platform",
    version="0.1.0",
)


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# In-memory storage for multiple tenders.
#
# Structure:
# {
#     "tender_id": {
#         "filename": "...",
#         "metadata": {...},
#         "requirements": [...],
#         "risks": [...],
#         "pages": [...],
#         "chunks": [...]
#     }
# }
tender_vector_store: dict[str, dict] = {}


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
        # 1. Extract PDF text and page information.
        extracted = extract_pdf_text(file_bytes)

        pages = extracted["pages"]

        # 2. Extract tender intelligence.
        intelligence = extract_tender_intelligence(pages)

        # 3. Create source-aware chunks.
        chunks = create_chunks(
            pages=pages,
            source_filename=file.filename,
        )

        # 4. Generate embeddings for semantic retrieval.
        embedded_chunks = embed_chunks(chunks)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # 5. Generate a unique ID for this tender.
    tender_id = f"tender_{uuid.uuid4().hex[:8]}"

    # 6. Store the tender independently.
    tender_vector_store[tender_id] = {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": len(file_bytes),
        "page_count": extracted["page_count"],
        "text_length": len(extracted["text"]),
        "metadata": intelligence["metadata"],
        "requirements": intelligence["requirements"],
        "risks": intelligence["risks"],
        "pages": pages,
        "chunks": embedded_chunks,
    }

    return {
        "tender_id": tender_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": len(file_bytes),
        "page_count": extracted["page_count"],
        "text_length": len(extracted["text"]),
        "metadata": intelligence["metadata"],
        "requirements": intelligence["requirements"],
        "risks": intelligence["risks"],
        "chunk_count": len(embedded_chunks),
    }


@app.get("/tenders")
async def list_tenders():

    tenders = []

    for tender_id, tender in tender_vector_store.items():
        tenders.append(
            {
                "tender_id": tender_id,
                "filename": tender["filename"],
                "page_count": tender["page_count"],
                "chunk_count": len(tender["chunks"]),
            }
        )

    return {
        "count": len(tenders),
        "tenders": tenders,
    }


@app.post("/search")
async def search_tender(
    tender_id: str,
    query: str,
    top_k: int = 5,
):

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    if tender_id not in tender_vector_store:
        raise HTTPException(
            status_code=404,
            detail="Tender not found.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )

    tender = tender_vector_store[tender_id]

    results = search_chunks(
        query=query,
        chunks=tender["chunks"],
        top_k=top_k,
    )

    return {
        "tender_id": tender_id,
        "query": query,
        "results": results,
    }


@app.post("/ask")
async def ask_tender(
    tender_id: str,
    question: str,
    top_k: int = 3,
):

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if tender_id not in tender_vector_store:
        raise HTTPException(
            status_code=404,
            detail="Tender not found.",
        )

    if top_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0.",
        )

    tender = tender_vector_store[tender_id]

    try:
        result = answer_question(
            question=question,
            chunks=tender["chunks"],
            top_k=top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "tender_id": tender_id,
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
    }