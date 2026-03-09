import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

SYMBOL = os.getenv("SYMBOL", "AAPL")
try:
    MAX_TRADE_AMOUNT = float(os.getenv("MAX_TRADE_AMOUNT", "1000"))
except ValueError:
    raise ValueError("MAX_TRADE_AMOUNT must be a valid number")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
