import os
try:
    from dotenv import load_dotenv
except ImportError:  # Allows dependency-light tests before full app installation.
    def load_dotenv() -> bool:
        return False

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 180
CHUNK_OVERLAP = 30
RETRIEVAL_K = 8
HYBRID_ALPHA = 0.45
WORD_BUDGET = 700
MAX_CONTEXT_CHUNKS = 5
