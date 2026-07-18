"""
Utilidad para cargar documentos y construir vector stores (uno por dominio).

Cada agente RAG (HR, Tech, Finance) tiene su propia colección de documentos:
un único archivo .txt con la base de conocimiento de esa área.

Este módulo se encarga de:
1. Leer el archivo de texto del dominio.
2. Dividirlo en chunks pequeños con RecursiveCharacterTextSplitter.
3. Generar embeddings con OpenAI y guardarlos en un vector store FAISS en memoria.

Nota: para mantener el proyecto simple, el vector store se reconstruye cada vez
que se corre el sistema (no se persiste en disco). Para un caso real en
producción, conviene guardar el índice FAISS con `vectorstore.save_local(...)`
y cargarlo con `FAISS.load_local(...)` para no re-generar embeddings cada vez.
"""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

CHUNK_SIZE = 250
CHUNK_OVERLAP = 30


def build_vector_store(doc_path: str) -> FAISS:
    """Carga un .txt, lo divide en chunks y construye un vector store FAISS."""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"No se encontró el documento: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    print(f"[vectorstore] {os.path.basename(doc_path)}: {len(chunks)} chunks generados")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore
