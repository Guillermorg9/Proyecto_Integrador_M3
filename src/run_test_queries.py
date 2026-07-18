"""
Punto de entrada para testeos de correcto enrutamiento del agente orquestador.

Flujo del sistema:
1. Carga variables de entorno (OpenAI + Langfuse) desde .env.
2. Construye los vector stores y los 3 agentes RAG especializados
   (HR, Tech, Finance), cada uno con su propia colección de documentos.
3. Crea el Orquestador, que clasifica la intención de cada consulta y la
   enruta al agente correcto.
4. Corre las consultas de prueba definidas en test_queries.json y muestra
   los resultados (categoría detectada vs esperada, y la respuesta generada).
"""

from dotenv import load_dotenv

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.utils import build_agents, load_test_queries
from src.agents.orchestrator import Orchestrator

load_dotenv()
langfuse = get_client()
langfuse_handler = CallbackHandler()


def main():
    agents = build_agents()
    orchestrator = Orchestrator(agents, langfuse_handler=langfuse_handler)

    test_queries = load_test_queries()
    correct = 0

    print("\n" + "=" * 70)
    print("EJECUTANDO CONSULTAS DE PRUEBA")
    print("=" * 70)

    for item in test_queries:
        query = item["query"]
        expected = item["expected_category"]

        result = orchestrator.route(query)
        is_correct = result["category"] == expected
        correct += int(is_correct)

        print(f"\nConsulta: {query}")
        print(
            f"Categoría esperada: {expected} | Categoría detectada: {result['category']} "
            f"{'✅' if is_correct else '❌'}"
        )
        print(f"Respuesta: {result['answer']}")

    print("\n" + "=" * 70)
    print(f"Precisión de routing: {correct}/{len(test_queries)}")
    print("=" * 70)

    if langfuse:
        try:
            langfuse.flush()
            print("\nTraces enviados a Langfuse.")
        except Exception as e:
            print(f"\nNo se pudieron enviar traces a Langfuse: {e}")


if __name__ == "__main__":
    main()
