EXPECTED_SN_VERSION = "004"  # Standard Notes backup format version
EMBEDDINGS_MODEL_NAME = "bge-m3"  # Ollama model for embeddings
MODEL_NAME = "qwen3.5:9b"  # Ollama model for LLM responses
TOKEN_WARN_RATIO = 0.75  # warn when the conversation estimate crosses this fraction
CONTEXT_WINDOW = 8192  # answer LLM's num_ctx; denominator for the context gauge
# Above this much entry text, a summary is built via map-reduce instead of one pass.
# ~45k chars ≈ 11k tokens, comfortably inside the summarizer's num_ctx=16384.
SUMMARY_CHAR_BUDGET = 45000
SEARCH_K = 10  # entries to retrieve for a point lookup (entries are tiny)
PERSIST_DIR = "./diary_vector_db"  # where to write the vector DB
