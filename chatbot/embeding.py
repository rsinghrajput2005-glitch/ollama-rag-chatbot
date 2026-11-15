from langchain_ollama import OllamaEmbeddings

def embedding_model():
    embeddings = OllamaEmbeddings(
    model='mxbai-embed-large'
    )
    
    return embeddings