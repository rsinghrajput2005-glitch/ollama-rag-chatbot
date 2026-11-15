def load_pdf(path):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path=path)

    docs = loader.load()

    return docs