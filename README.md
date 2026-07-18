# Sistema Multiagente de Soporte (Orquestador + 3 RAG especializados)

Sistema de soporte interno que enruta consultas de empleados hacia el agente correcto según la intención detectada. Un **Orquestador** clasifica cada consulta y la deriva a uno de tres agentes **RAG** especializados:

- 🧑‍💼 **RH (HR)** — políticas de personal, nómina, prestaciones, vacaciones, etc.
- 💰 **Finanzas** — reembolsos, viáticos, facturación, presupuestos, etc.
- 💻 **TI (Tech)** — hardware, software, periféricos, accesos, soporte técnico, etc.

## Arquitectura

```
Usuario ──> Orquestador (clasifica intención) ──┬──> Agente RAG RH
                                                ├──> Agente RAG Finanzas
                                                └──> Agente RAG TI
```

Cada agente RAG tiene su propia colección de documentos (vector store), lo que le permite responder únicamente sobre su dominio de especialidad.

## Requisitos previos

- Python 3.11+ (o la versión especificada en `pyproject.toml`)
- [`uv`](https://docs.astral.sh/uv/) instalado como gestor de paquetes
- Una API key de OpenAI válida
- Credenciales de [Langfuse](https://langfuse.com/) si se quiere trazabilidad/observabilidad de las llamadas

## Instalación

1. Clona el repositorio y entra a la carpeta del proyecto:

   ```bash
   git clone <url-del-repo>
   cd <nombre-del-proyecto>
   ```

2. Instala las dependencias con `uv` (crea el entorno virtual automáticamente a partir de `pyproject.toml`):

   ```bash
   uv sync
   ```

3. Crea un archivo `.env` en la raíz del proyecto con tus variables de entorno:

   ```env
   OPENAI_API_KEY=sk-...

   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

## Estructura del proyecto (relevante)

```
.
├── src/
│   ├── multi_agent_system.py     # Punto de entrada interactivo
│   ├── run_test_queries.py       # Punto de entrada para test de enrutamiento
│   └── ...                       # Orquestador, agentes RAG, vector stores, etc.
├── test_queries.json             # Consultas de prueba con su categoría esperada
```

## Cómo probar el proyecto

Hay dos formas de correrlo:

### 1. Modo interactivo (consulta única)

Ejecuta una consulta puntual y observa cómo el orquestador la clasifica y enruta:

```bash
uv run python -m src.multi_agent_system "que pasa si quiero pedir un periferico nuevo, un teclado?"
```

**Qué hace:**

1. Carga las variables de entorno (OpenAI + Langfuse) desde `.env`.
2. Construye los vector stores y los 3 agentes RAG especializados (HR, Tech, Finance), cada uno con su propia colección de documentos.
3. Crea el Orquestador, que clasifica la intención de la consulta y la enruta al agente correcto.

En el ejemplo de arriba, se esperaría que el orquestador detecte la categoría **TI** y responda usando el agente RAG de Tech.


### 2. Modo test (enrutamiento automatizado)

Corre un set de consultas predefinidas y valida que el orquestador las enrute a la categoría esperada:

```bash
uv run python -m src.run_test_queries
```

**Qué hace:**

1. Carga las variables de entorno (OpenAI + Langfuse) desde `.env`.
2. Construye los vector stores y los 3 agentes RAG especializados (HR, Tech, Finance), cada uno con su propia colección de documentos.
3. Crea el Orquestador, que clasifica la intención de cada consulta y la enruta al agente correcto.
4. Corre las consultas definidas en `test_queries.json` y muestra los resultados: categoría detectada vs. categoría esperada, y la respuesta generada por el agente correspondiente.

#### Formato de `test_queries.json`

Cada entrada define una consulta y la categoría que se espera que el orquestador detecte:

```json
[
  {
    "query": "¿Cuándo se paga la nómina y qué hago si hay un error en mi recibo?",
    "expected_category": "hr"
  }
]
```

Para agregar nuevos casos de prueba, solo añade objetos con `query` y `expected_category` (`hr`, `finance` o `tech`) al arreglo