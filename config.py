"""Model configuration shared by the app and seed script."""

GROQ_MODEL = "openai/gpt-oss-120b"
VECTOR_DIMENSIONS = 1024
CONTEXT_TOKEN_BUDGET = 700
RAG_TOP_K = 4
MAX_LLM_CALLS_PER_SESSION = 15
