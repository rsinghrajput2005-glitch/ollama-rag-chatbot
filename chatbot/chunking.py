from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=75)
    chunks = splitter.split_documents(docs)

    return chunks