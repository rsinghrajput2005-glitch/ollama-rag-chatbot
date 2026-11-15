from langchain_ollama import OllamaEmbeddings

def embedding_model():
    return OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )