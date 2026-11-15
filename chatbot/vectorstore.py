from langchain_chroma import Chroma
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "new_chroma_db"

def vector_database(chunks, embeddings):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="test_collection",
        persist_directory=str(DB_DIR)
    )

    return vectorstore