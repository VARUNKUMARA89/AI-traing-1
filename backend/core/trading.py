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
        try:
            response = self.client.get_positions()
            logger.debug(f"Raw positions response: {response}")
            
            # Convert response to list if it's not already
            if isinstance(response, str):
                # Handle empty or error response
                if "No positions" in response:
                    return []
                logger.warning(f"Unexpected string response: {response}")
                return []
                
            # If response is a dictionary, check for positions data
            if isinstance(response, dict):
                positions = response.get("positions", [])
                if isinstance(positions, list):
                    return positions
                return []
                
            # If response is already a list
            if isinstance(response, list):
                return response
                
            logger.warning(f"Unexpected response type: {type(response)}")
            return []
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
        
    def get_pnl(self) -> Dict:
        """Get current P&L"""
        try:
            positions = self.get_positions()
            
            if not positions:
                return {
                    "total_pnl": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "positions_count": 0
                }
            
            total_pnl = 0.0
            for position in positions:
                # Try different possible PNL field names
                pnl = position.get("pnl", 
                      position.get("unrealizedPnL",
                      position.get("realizedPnL", 0)))
                
                if isinstance(pnl, str):
                    try:
                        pnl = float(pnl.replace(',', ''))
                    except (ValueError, AttributeError):
                        pnl = 0.0
                elif isinstance(pnl, (int, float)):
                    pnl = float(pnl)
                else:
                    pnl = 0.0
                    
                total_pnl += pnl
            
            return {
                "total_pnl": total_pnl,
                "timestamp": datetime.now().isoformat(),
                "positions_count": len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return {
                "total_pnl": 0.0,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "positions_count": 0
            }