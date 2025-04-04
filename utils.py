import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


def load_api_keys():
    """Load environment variables and return API keys."""
    load_dotenv()
    return {
        "huggingface": os.getenv("HUGGINGFACEHUB_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY")
    }


def create_db_engine():
    """Create and return a SQLite database engine."""
    return create_engine("sqlite:///./Database/hotel_database.db")


