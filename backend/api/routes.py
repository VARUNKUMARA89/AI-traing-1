from fastapi import APIRouter, HTTPException
from core.trading import TradingEngine
from core.config import settings
from loguru import logger

api_router = APIRouter()
trading_engine = TradingEngine()

@api_router.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "running",
        "version": settings.VERSION,
        "trading_active": trading_engine.is_active
    }

@api_router.get("/positions")
async def get_positions():
    """Get current positions"""
    try:
        return trading_engine.get_positions()
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/pnl")
async def get_pnl():
    """Get current P&L"""
    try:
        return trading_engine.get_pnl()
    except Exception as e:
        logger.error(f"Error fetching P&L: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/start")
async def start_trading():
    """Start the trading engine"""
    try:
        await trading_engine.start()
        return {"status": "Trading started"}
    except Exception as e:
        logger.error(f"Error starting trading engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/stop")
async def stop_trading():
    """Stop the trading engine"""
    try:
        await trading_engine.stop()
        return {"status": "Trading stopped"}
    except Exception as e:
        logger.error(f"Error stopping trading engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))