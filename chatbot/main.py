import os
import shutil

from document_loader import load_pdf
from chunking import splitter
from embeding import embedding_model
from vectorstore import vector_database
from retriever import retrieve_docs
from model import final_output

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Uploaded PDFs are saved here (next to main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def last(path, question):
    docs = load_pdf(path)
    chunks = splitter(docs)
    embiddings = embedding_model()
    vectorstore = vector_database(chunks, embiddings)
    filtered_docs = retrieve_docs(question, vectorstore)
    final = final_output(filtered_docs, question)
    return final


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    filename = os.path.basename(file.filename)  # strip any folder parts
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": filename}


@app.post("/get_answer")
def chatbot(question: str, filename: str):
    path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF not found. Upload it first.")

    answer = last(path=path, question=question)
    return answer
