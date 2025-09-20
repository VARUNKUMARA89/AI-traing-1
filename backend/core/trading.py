from typing import Dict, List, Optional
from datetime import datetime
import asyncio
from loguru import logger
from dhanhq import dhanhq
from core.config import settings
from data.market_data import MarketDataManager
from strategies.ai_strategy import AITradingStrategy

class TradingEngine:
    """Main trading engine that coordinates market data, strategies, and execution"""
    
    def __init__(self):
        self.is_active = False
        self.client = dhanhq(
            client_id=settings.DHAN_CLIENT_ID,
            access_token=settings.DHAN_ACCESS_TOKEN
        )
        # Initialize the same client for market data streaming
        self.market_feed = self.client
        self.market_data = MarketDataManager(self.client)
        self.strategy = AITradingStrategy()
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the trading engine"""
        if self.is_active:
            return
            
        self.is_active = True
        self._task = asyncio.create_task(self._run_trading_loop())
        logger.info("Trading engine started")
        
    async def stop(self):
        """Stop the trading engine"""
        if not self.is_active:
            return
            
        self.is_active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Trading engine stopped")
        
    async def _run_trading_loop(self):
        """Main trading loop"""
        try:
            while self.is_active:
                # Get latest market data
                data = await self.market_data.get_latest_data()
                
                # Generate trading signals
                signals = self.strategy.generate_signals(data)
                
                # Execute trades based on signals
                for signal in signals:
                    await self._execute_trade(signal)
                    
                await asyncio.sleep(1)  # Throttle loop
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.is_active = False
            raise
            
    async def _execute_trade(self, signal: Dict):
        """Execute a trade based on the signal"""
        try:
            # Place order via DhanHQ client
            order = self.client.place_order(
                security_id=signal["security_id"],
                exchange_segment=signal["exchange"],
                transaction_type=signal["side"],
                quantity=signal["quantity"],
                order_type="MARKET",
                product_type="INTRADAY",
                price=0  # Market order
            )
            logger.info(f"Order placed: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise
            
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return self.client.get_positions()
        
    def get_pnl(self) -> Dict:
        """Get current P&L"""
        positions = self.get_positions()
        total_pnl = sum(float(pos.get("pnl", 0)) for pos in positions)
        return {
            "total_pnl": total_pnl,
            "timestamp": datetime.now().isoformat()
        }