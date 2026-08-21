# from extraction import extract_pdf_text
# from processing import create_chunks, embed_chunks
# from rag import answer_question

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PDF_PATH = "data/TenderIQ_Test_Tender_02.pdf"


# with open(PDF_PATH, "rb") as file:
#     pdf_bytes = file.read()


# # 1. Extract PDF
# extracted = extract_pdf_text(pdf_bytes)

# pages = extracted["pages"]

# # 2. Create chunks
# chunks = create_chunks(
#     pages=pages,
#     source_filename="TenderIQ_Test_Tender_02.pdf",
# )

# # 3. Generate embeddings
# embedded_chunks = embed_chunks(chunks)

# print(f"Pages: {len(pages)}")
# print(f"Chunks: {len(chunks)}")
# print(f"Embedded chunks: {len(embedded_chunks)}")


# # 4. Ask TenderIQ a question
# result = answer_question(
#     question="What are the key risks and constraints?, What is the planned construction period?, What documents are required for submission?, What are the mandatory eligibility requirements?, Does the tender require a performance bond?",
#     chunks=embedded_chunks,
#     top_k=5,
# )


# print("\nANSWER:")
# print(result["answer"])

# print("\nSOURCES:")
# for source in result["sources"]:
#     print(source)



import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extraction import extract_pdf_text
from processing import create_chunks, embed_chunks
from rag import answer_question


PDF_PATH = "data/TenderIQ_Test_Tender_02.pdf"
SOURCE_FILENAME = "TenderIQ_Test_Tender_02.pdf"


# 1. Load PDF
with open(PDF_PATH, "rb") as file:
    pdf_bytes = file.read()


# 2. Extract PDF text
extracted = extract_pdf_text(pdf_bytes)
pages = extracted["pages"]


# 3. Create chunks
chunks = create_chunks(
    pages=pages,
    source_filename=SOURCE_FILENAME,
)


# 4. Generate embeddings
embedded_chunks = embed_chunks(chunks)


print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"Embedded chunks: {len(embedded_chunks)}")


# 5. Test questions
questions = [
    "What are the key risks and constraints?",
    "What is the planned construction period?",
    "What documents are required for submission?",
    "What are the mandatory eligibility requirements?",
    "Does the tender require a performance bond?",
]


# 6. Ask TenderIQ each question
for question in questions:
    result = answer_question(
        question=question,
        chunks=embedded_chunks,
        top_k=3,
    )

    print("\n" + "=" * 80)
    print(f"QUESTION:\n{question}")

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(source)