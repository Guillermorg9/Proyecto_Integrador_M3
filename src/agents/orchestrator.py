"""
Agente Orquestador.

Se encarga de dos cosas:
1. Clasificar la intención de la consulta del usuario (hr / tech / finance).
2. Enrutar la consulta al agente RAG especializado correspondiente.

La clasificación se hace con un LLM (gpt-4o-mini) usando un prompt simple que
le pide responder únicamente con una palabra (la categoría). Esto es un
"routing por LLM": más flexible que reglas por palabras clave, porque entiende
la intención aunque la consulta esté redactada de formas distintas.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

VALID_CATEGORIES = ["hr", "tech", "finance"]

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un clasificador de intención para un sistema de soporte al cliente. "
            "Clasifica la consulta del usuario en EXACTAMENTE una de estas categorías:\n"
            "- hr: preguntas sobre recursos humanos, vacaciones, beneficios, nómina, "
            "onboarding, licencias, evaluaciones de desempeño, políticas laborales.\n"
            "- tech: preguntas sobre soporte técnico o IT, accesos, contraseñas, VPN, "
            "hardware, software, seguridad de la información.\n"
            "- finance: preguntas sobre finanzas, reembolsos, facturación, presupuestos, "
            "pagos a proveedores, viáticos, tarjeta corporativa.\n\n"
            "Responde ÚNICAMENTE con una palabra: hr, tech o finance. "
            "Sin explicaciones, sin puntuación, sin mayúsculas.",
        ),
        ("human", "{query}"),
    ]
)


class Orchestrator:
    """Clasifica la intención y enruta hacia el agente especializado correcto."""

    def __init__(self, agents: dict, langfuse_handler=None):
        self.agents = agents
        self.classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.langfuse_handler = langfuse_handler

    def callback_config(self) -> dict:
        if self.langfuse_handler:
            return {"callbacks": [self.langfuse_handler]}
        return {}

    def classify_query(self, query: str) -> str:
        """Devuelve la categoría detectada (hr, tech o finance)."""
        chain = CLASSIFICATION_PROMPT | self.classifier_llm | StrOutputParser()
        result = chain.invoke({"query": query}, config=self.callback_config())
        category = result

        if category not in VALID_CATEGORIES:
            print(
                f"[orchestrator] categoría inesperada '{category}', usando fallback 'tech'"
            )
            category = "tech"
        return category

    def route(self, query: str) -> dict:
        """Clasifica la consulta y la delega al agente RAG correspondiente."""
        category = self.classify_query(query)
        agent = self.agents[category]

        result = agent.invoke(query, config=self.callback_config())

        return {
            "query": query,
            "category": category,
            "answer": result,
        }
