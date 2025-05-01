import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys and Secrets
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Solana Configuration
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOLANA_WEBSOCKET_URL = os.getenv("SOLANA_WEBSOCKET_URL", "wss://api.mainnet-beta.solana.com")
SOLANA_COMMITMENT = os.getenv("SOLANA_COMMITMENT", "confirmed")

# DUNE Token Configuration
DUNE_TOKEN_ADDRESS = os.getenv("DUNE_TOKEN_ADDRESS")
DUNE_TOKEN_DECIMALS = int(os.getenv("DUNE_TOKEN_DECIMALS", "9"))

# Scanner Configuration
MIN_MARKET_CAP_THRESHOLD = float(os.getenv("MIN_MARKET_CAP_THRESHOLD", "1000000"))  # 1M USD
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
MAX_TOKENS_TO_TRACK = int(os.getenv("MAX_TOKENS_TO_TRACK", "100"))

# Sentiment Analysis Configuration
SENTIMENT_ANALYSIS_BATCH_SIZE = int(os.getenv("SENTIMENT_ANALYSIS_BATCH_SIZE", "100"))
SENTIMENT_THRESHOLD_POSITIVE = float(os.getenv("SENTIMENT_THRESHOLD_POSITIVE", "0.7"))
SENTIMENT_THRESHOLD_NEGATIVE = float(os.getenv("SENTIMENT_THRESHOLD_NEGATIVE", "0.3"))

# Database Configuration (for future implementation)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/dune_ai.db")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))