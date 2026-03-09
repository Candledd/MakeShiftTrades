# MakeShiftTrades
Vibecoded Daytrading Bot

## Project structure

```
MakeShiftTrades/
├── .env.example      # Template for environment variables
├── .gitignore        # Python-specific git ignores
├── config.py         # Loads settings from .env
├── main.py           # Entry point — run this to start the bot
├── requirements.txt  # Python dependencies
├── setup.sh          # One-command bootstrap script
└── src/
    ├── __init__.py
    ├── trader.py     # Core Trader class — add your strategy here
    └── utils.py      # Shared helpers (logging, etc.)
```

## Quick start

**Prerequisites:** Python 3.8+ and Git.

```bash
# 1. Clone the repo
git clone https://github.com/Candledd/MakeShiftTrades.git
cd MakeShiftTrades

# 2. Bootstrap the virtual environment and install dependencies
bash setup.sh

# 3. Activate the virtual environment
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 4. Configure your environment variables
cp .env.example .env
# Edit .env and fill in your API keys and trading parameters

# 5. Run the bot
python main.py
```

## Configuration

All settings are loaded from a `.env` file (never committed to git).
Copy `.env.example` to `.env` and adjust the values:

| Variable           | Default | Description                          |
|--------------------|---------|--------------------------------------|
| `API_KEY`          | —       | Your broker API key                  |
| `API_SECRET`       | —       | Your broker API secret               |
| `SYMBOL`           | AAPL    | Ticker symbol to trade               |
| `MAX_TRADE_AMOUNT` | 1000    | Maximum USD amount per trade         |
| `DRY_RUN`          | true    | Set to `false` to place real orders  |

## Implementing your strategy

Open `src/trader.py` and fill in the `should_buy` and `should_sell` methods,
and update `get_price` with a real API call to your broker or data provider.

