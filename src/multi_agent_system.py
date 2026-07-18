"""
Punto de entrada principal del sistema multiagente de routing de soporte.

Flujo del sistema:
1. Carga variables de entorno (OpenAI + Langfuse) desde .env.
2. Construye los vector stores y los 3 agentes RAG especializados
   (HR, Tech, Finance), cada uno con su propia colección de documentos.
3. Crea el Orquestador, que clasifica la intención de cada consulta y la
   enruta al agente correcto.

"""

import json
import sys

from dotenv import load_dotenv

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.utils import build_agents
from src.agents.orchestrator import Orchestrator

load_dotenv()
langfuse = get_client()
langfuse_handler = CallbackHandler()


def main():
    query = " ".join(sys.argv[1:])

    if len(query) == 0:
        print("⚠️  Advertencia: no se ingresó ninguna consulta.", file=sys.stderr)
        print(
            'Uso: uv run python -m src.main "<pregunta del usuario>"', file=sys.stderr
        )
        sys.exit(1)

    agents = build_agents()
    orchestrator = Orchestrator(agents, langfuse_handler=langfuse_handler)

    result = orchestrator.route(query)

    print("\n" + "=" * 70)
    print("ASSISTANT RESPONSE")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 70)

    if langfuse:
        try:
            langfuse.flush()
            print("\nTraces enviados a Langfuse.")
        except Exception as e:
            print(f"\nNo se pudieron enviar traces a Langfuse: {e}")


if __name__ == "__main__":
    main()
