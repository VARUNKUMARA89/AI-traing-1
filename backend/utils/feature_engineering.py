from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import zscore
from loguru import logger
from utils.data_validation import validate_ohlcv, resample_ohlcv

def create_target(df: pd.DataFrame, 
                 horizon: int = 3, 
                 min_return: float = 0.002,
                 use_atr: bool = True) -> pd.Series:
    """
    Create classification target based on future returns
    
    Args:
        df: DataFrame with OHLCV data
        horizon: Number of periods to look ahead
        min_return: Minimum return threshold for signal (0.002 = 0.2%)
    
    Returns:
        Series with 1 (Buy), 0 (Sell), or np.nan (Hold)
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    target = np.where(future_return > min_return, 1,
                     np.where(future_return < -min_return, 0, np.nan))
    return pd.Series(target, index=df.index)

def calculate_features(df: pd.DataFrame,
                     resample_freq: Optional[str] = '5min',
                     add_lagged_features: bool = True,
                     add_ta_features: bool = True) -> pd.DataFrame:
    """
    Calculate technical and market microstructure features
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        resample_freq: If provided, resample data to this frequency first
        add_lagged_features: Whether to add lagged features
        add_ta_features: Whether to add technical analysis features
        
    Returns:
        DataFrame with features
    """
    # Validate and clean data
    df, warnings = validate_ohlcv(df)
    for warning in warnings:
        logger.warning(warning)
    
    # Resample if requested
    if resample_freq and isinstance(df.index, pd.DatetimeIndex):
        df = resample_ohlcv(df, freq=resample_freq)
    
    features = pd.DataFrame(index=df.index)
    
    # Price action features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    
    # Rolling statistics
    for window in [5, 20, 50]:
        # Returns
        features[f'volatility_{window}'] = features['returns'].rolling(window).std()
        features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
        
        # Price
        features[f'close_sma_{window}'] = df['close'].rolling(window).mean()
        features[f'high_low_range_{window}'] = (
            df['high'].rolling(window).max() - df['low'].rolling(window).min()
        ) / df['close']
        
        # Volume
        features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
        features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
    
    if add_ta_features:
        # Technical indicators
        features['rsi'] = df.ta.rsi(length=14)
        macd = df.ta.macd()
        features['macd'] = macd['MACD_12_26_9']
        features['macd_signal'] = macd['MACDs_12_26_9']
        features['macd_hist'] = macd['MACDh_12_26_9']
        
        # Volatility indicators
        features['atr'] = df.ta.atr(length=14)
        features['atr_pct'] = features['atr'] / df['close']
        
        # Trend indicators
        adx = df.ta.adx()
        features['adx'] = adx['ADX_14']
        features['di_plus'] = adx['DMP_14']
        features['di_minus'] = adx['DMN_14']
        
        # Volume analysis
        features['obv'] = df.ta.obv()
        features['relative_volume'] = df['volume'] / features['volume_sma_20']
        
        # Additional indicators
        features['cci'] = df.ta.cci()
        bb = df.ta.bbands()
        features['bb_upper'] = bb['BBU_5_2.0']
        features['bb_lower'] = bb['BBL_5_2.0']
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / df['close']
        
    # Lagged features
    if add_lagged_features:
        for col in ['returns', 'log_returns', 'volume']:
            if col in features:
                for lag in [1, 2, 3, 5]:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)
                    
        # Return acceleration
        features['returns_acc'] = features['returns'] - features['returns'].shift(1)
    
    # VWAP and deviation
    features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    features['vwap_diff'] = (df['close'] - features['vwap']) / features['vwap']
    
    # Time-based features (assuming timestamp index)
    if isinstance(df.index, pd.DatetimeIndex):
        features['hour'] = df.index.hour
        features['minute'] = df.index.minute
        features['time_of_day'] = features['hour'] + features['minute'] / 60
        
    # Z-score normalization for non-categorical features
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['hour', 'minute', 'time_of_day']:
            features[f'{col}_zscore'] = zscore(features[col], nan_policy='omit')
    
    return features.fillna(method='ffill').fillna(0)