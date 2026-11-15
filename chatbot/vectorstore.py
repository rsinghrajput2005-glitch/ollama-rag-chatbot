from langchain_chroma import Chroma

def vector_database(chunks, embeddings):
    vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    collection_name="test_collection",
    persist_directory="./new_chroma_db"
    )

    return vectorstore