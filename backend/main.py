from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
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
)


app = FastAPI(
    title="TenderIQ",
    description="AI-powered construction tender intelligence platform",
    version="0.1.0",
)


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

tender_vector_store: list[dict] = []


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
        # Extract PDF text and page information.
        extracted = extract_pdf_text(file_bytes)

        pages = extracted["pages"]

        # Extract tender intelligence.
        intelligence = extract_tender_intelligence(pages)

        # Create source-aware chunks.
        chunks = create_chunks(
            pages=pages,
            source_filename=file.filename,
        )

        # Generate embeddings for semantic retrieval.
        embedded_chunks = embed_chunks(chunks)

        # Replace the current tender in memory.
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
async def search_tender(
    query: str,
    top_k: int = 5,
):

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