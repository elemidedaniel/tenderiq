from pathlib import Path

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile


app = FastAPI(
    title="TenderIQ",
    description="AI-powered construction tender intelligence platform",
    version="0.1.0",
)


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def extract_pdf_text(file_bytes: bytes) -> dict:
    """
    Extract text from every page of a PDF.

    Returns page-level text so that we preserve document structure
    for the RAG pipeline we will build on Day 2.
    """

    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError("The uploaded file is not a valid PDF.") from exc

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()

            pages.append(
                {
                    "page": page_number,
                    "text": text,
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
        extracted = extract_pdf_text(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "page_count": extracted["page_count"],
        "text_length": len(extracted["text"]),
        "pages": extracted["pages"],
    }