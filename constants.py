EXPECTED_SN_VERSION = "004"  # Standard Notes backup format version
EMBEDDINGS_MODEL_NAME = "bge-m3"  # Ollama model for embeddings
MODEL_NAME = "qwen3.5:9b"  # Ollama model for LLM responses
TOKEN_WARN_RATIO = 0.75  # warn when the conversation estimate crosses this fraction
CONTEXT_WINDOW = 8192  # query LLM's num_ctx (history budget); the context-gauge denom
# Above this much entry text, generation uses map-reduce instead of a single pass.
# ~45k chars ≈ 11k tokens, comfortably inside the generation LLM's num_ctx=16384.
SINGLE_PASS_BUDGET = 45000
# Filtered scopes with at most this many entries are fetched in FULL (no top-K cap).
FETCH_ALL_MAX = 50
SEARCH_K = 10  # entries to retrieve for a point lookup (entries are tiny)
# Min rapidfuzz partial_ratio (0-100) for a keyword/entity match. Fuzzy (not exact) so
# it survives Czech declension + casing; verbatim mentions score 100. Tune if it
# over/under-matches on real entries.
FUZZY_MATCH_THRESHOLD = 80
PERSIST_DIR = "./diary_vector_db"  # where to write the vector DB

# Multilingual synonyms for the fixed Czech diary tags. Fed to the extraction LLM
# (so e.g. "skiing" maps to lyže + skialp) and used by the regex fallback. A term
# listed under several tags (e.g. "skiing") selects all of them. Verify/extend as
# tags change; tags not listed here still work, just without cross-lingual hints.
TAG_ALIASES: dict[str, list[str]] = {
    "běh": ["running", "run", "běhání"],
    "plavání": ["swimming", "swim"],
    "diplomka": ["thesis", "master's thesis", "diplomová práce"],
    "státnice": ["state exams", "final exams", "státní zkoušky"],
    "cvičení": ["exercise", "workout"],
    "fitko": ["gym", "weights", "posilovna", "fitness", "workout"],
    "lyže": ["skiing", "ski", "downhill skiing", "lyžování"],
    "skialp": ["skiing", "ski touring", "ski mountaineering", "skialpinismus"],
    "mtb": ["mountain biking", "biking", "cycling", "horské kolo", "kolo"],
    "turistika": ["hiking", "trekking", "túra"],
    "lezení": ["climbing", "bouldering", "via ferrata"],
    "koloběžka": ["push bike"],
}

MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
