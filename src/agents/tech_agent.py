"""
Agente RAG especializado en Finanzas.

Responde preguntas usando SOLO la base de conocimiento en data/tech_docs/tech_knowledge.txt, para evitar que el modelo invente políticas que no existen.
"""

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.vector_store import build_vector_store

DOC_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "tech_docs"
    / "tech_knowledge.txt"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente Soporte Técnico (IT) de la empresa. "
            "Responde la pregunta del usuario usando SOLO la información del contexto. "
            "Si el contexto no contiene la respuesta, dilo claramente en vez de inventar una política financiera que no existe.\n\n",
        ),
        ("human", "Contexto:\n{context}\n\nPregunta:\n{question}"),
    ]
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_tech_agent() -> Runnable:
    """Construye y devuelve la chain RAG del agente de IT."""
    vectorstore = build_vector_store(str(DOC_PATH))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain
