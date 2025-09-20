from dhanhq import dhanhq
import os
from dotenv import load_dotenv
from loguru import logger

def test_dhan_connection():
    """Test connection to Dhan API and fetch basic account information"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get credentials from environment
        client_id = os.getenv("DHAN_CLIENT_ID")
        access_token = os.getenv("DHAN_ACCESS_TOKEN")
        
        if not client_id or not access_token:
            logger.error("DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN not found in .env file")
            return False
            
        # Initialize DhanHQ client
        dhan = dhanhq(client_id=client_id, access_token=access_token)
        
        # Test connection by fetching holdings
        holdings = dhan.get_holdings()
        logger.info(f"Successfully connected to Dhan API")
        logger.info(f"Holdings: {holdings}")
        
        # Test market data access
        logger.info("\nFetching market information:")
        try:
            # Get order book
            order_book = dhan.get_order_list()
            logger.info(f"Order Book: {order_book}")
            
            # Get positions
            positions = dhan.get_positions()
            logger.info(f"Positions: {positions}")
            
            # Get quote data for testing
            symbols = {'NSE': ['INFY']}  # Example symbol
            quotes = dhan.quote_data(symbols)
            logger.info(f"Quote Data: {quotes}")
            
        except Exception as e:
            logger.warning(f"Could not fetch market data: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Dhan API: {e}")
        return False

if __name__ == "__main__":
    test_dhan_connection()