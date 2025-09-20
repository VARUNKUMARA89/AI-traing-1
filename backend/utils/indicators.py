import pandas as pd
import numpy as np

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators as features for ML model"""
    df = df.copy()
    
    # RSI
    def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # MACD
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    # Bollinger Bands
    def calculate_bollinger_bands(data: pd.Series, window: int = 20):
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        return upper, middle, lower
    
    # ATR (Average True Range)
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()
    
    # ADX (Average Directional Index)
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        tr = calculate_atr(high, low, close, window)
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(window).mean() / tr)
        neg_di = 100 * (neg_dm.rolling(window).mean() / tr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window).mean()
        return adx
    
    # VWAP (Volume Weighted Average Price)
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        v = df['volume']
        h = df['high']
        l = df['low']
        c = df['close']
        
        typical_price = (h + l + c) / 3
        vwap = (typical_price * v).cumsum() / v.cumsum()
        return vwap
    
    # Calculate all features
    df['rsi_14'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    df['adx'] = calculate_adx(df['high'], df['low'], df['close'])
    df['vwap'] = calculate_vwap(df)
    
    return df