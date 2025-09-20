# AI-Powered Intraday Trading System

This is a sophisticated intraday trading system that uses AI/ML models for market prediction and automated trading through the Dhan API. The system includes real-time market data processing, multiple trading strategies, backtesting capabilities, and a web-based dashboard for monitoring and control.

## Features

- Real-time market data processing via DhanHQ API
- AI/ML-based trading strategies
- Advanced backtesting engine
- Web-based monitoring dashboard
- Automated order execution
- Performance analytics and logging
- Multi-instrument support
- Telegram notifications

## Setup

1. Clone this repository
2. Install Python 3.10+ and Node.js 16+
3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

6. Set up the frontend:
```bash
cd frontend
npm install
```

## Usage

1. Start the backend server:
```bash
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Access the dashboard at http://localhost:3000

## Project Structure

```
├── backend/
│   ├── api/            # FastAPI routes
│   ├── core/           # Trading engine core
│   ├── data/           # Market data handling
│   ├── strategies/     # Trading strategies
│   ├── backtest/       # Backtesting engine
│   └── utils/          # Helper functions
├── frontend/           # React dashboard
├── models/             # Saved ML models
├── notebooks/          # Jupyter notebooks
└── tests/             # Unit tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT