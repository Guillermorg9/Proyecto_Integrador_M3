import json
from pathlib import Path

from src.agents.finance_agent import build_finance_agent
from src.agents.hr_agent import build_hr_agent
from src.agents.tech_agent import build_tech_agent

TEST_QUERIES_PATH = Path(__file__).resolve().parent.parent / "test_queries.json"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 30


def build_agents() -> dict:
    """Construye los 3 agentes RAG especializados."""
    print("Construyendo vector stores y agentes RAG...\n")
    return {
        "hr": build_hr_agent(),
        "tech": build_tech_agent(),
        "finance": build_finance_agent(),
    }


def load_test_queries() -> list:
    """Carga el archivo test_queries.json."""
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
