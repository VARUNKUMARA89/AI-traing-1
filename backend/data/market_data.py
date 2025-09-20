from typing import Dict, List, Optional
import asyncio
import pandas as pd
from dhanhq import dhanhq
from loguru import logger

class MarketDataManager:
    """Handles market data collection and processing"""
    
    def __init__(self, client: dhanhq):
        self.client = client
        self.data_buffer: Dict[str, List] = {}  # Buffer for each instrument
        
    async def start_feed(self, instruments: List[Dict]):
        """Start the market data feed for given instruments"""
        if not instruments:
            return
            
        try:
            # Format instruments for streaming - group by exchange
            exchange_instruments = {}
            for instr in instruments:
                exchange = instr["exchange"]
                security_id = str(instr["security_id"])
                if exchange not in exchange_instruments:
                    exchange_instruments[exchange] = []
                exchange_instruments[exchange].append(security_id)
            
            # Start real-time data streaming
            self.client.ticker_data(exchange_instruments)
            logger.info(f"Started streaming for {len(instruments)} instruments")
        except Exception as e:
            logger.error(f"Error starting market data feed: {e}")
        asyncio.create_task(self._process_feed())
        logger.info(f"Started market feed for {len(instruments)} instruments")
        
    async def _process_feed(self):
        """Process incoming market data"""
        try:
            while True:
                data = self.feed.get_data()
                if data:
                    security_id = data["security_id"]
                    self.data_buffer.setdefault(security_id, []).append({
                        "timestamp": pd.Timestamp.now(),
                        "ltp": data["ltp"],
                        "volume": data["volume"],
                        "bid": data.get("best_bid", 0),
                        "ask": data.get("best_ask", 0)
                    })
                    # Keep only recent data
                    self.data_buffer[security_id] = self.data_buffer[security_id][-1000:]
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing market feed: {e}")
            raise
            
    async def get_latest_data(self) -> Dict[str, pd.DataFrame]:
        """Get latest market data for all instruments"""
        result = {}
        for security_id, buffer in self.data_buffer.items():
            if buffer:
                df = pd.DataFrame(buffer)
                df.set_index("timestamp", inplace=True)
                result[security_id] = df
        return result
        
    def get_historical_data(self, security_id: str, exchange: str, 
                          from_date: str, to_date: str) -> pd.DataFrame:
        """Get historical data for backtesting"""
        try:
            data = self.client.historical_data(
                security_id=security_id,
                exchange=exchange,
                from_date=from_date,
                to_date=to_date
            )
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise