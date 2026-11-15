import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .document_loader import load_pdf
from .chunking import splitter
from .embedding import embedding_model
from .vectorstore import vector_database
from .retriever import retrieve_docs
from .model import final_output


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model once
embeddings = embedding_model()

# Store vector databases for uploaded PDFs
vectorstores = {}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    filename = os.path.basename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process PDF only once during upload
    docs = load_pdf(save_path)
    chunks = splitter(docs)

    vectorstore = vector_database(chunks, embeddings)

    vectorstores[filename] = vectorstore

    return {"filename": filename, "message": "PDF uploaded and processed successfully"}


@app.post("/get_answer")
def chatbot(question: str, filename: str):

    vectorstore = vectorstores.get(filename)

    if vectorstore is None:
        raise HTTPException(
            status_code=404,
            detail="PDF not found or not processed. Upload it first."
        )

    filtered_docs = retrieve_docs(question, vectorstore)

    answer = final_output(filtered_docs, question)

    return {"answer": answer}