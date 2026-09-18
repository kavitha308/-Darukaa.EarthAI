import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "Darukaa.Earth: AI Biodiversity Intelligence System"
    VERSION: str = "1.0.0"
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Data Paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DOCS_DIR: Path = DATA_DIR / "raw_docs"
    SEED_KNOWLEDGE_PATH: Path = DATA_DIR / "seed_knowledge.json"
    VECTOR_DB_DIR: Path = BASE_DIR / os.getenv("VECTOR_DB_DIR", "data/chroma_db")
    
    # Model Specs
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # Reasoning Thresholds
    MIN_METRICS_FOR_REASONING: int = 3
    
settings = Settings()
