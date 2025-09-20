from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger

def validate_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and clean OHLCV data
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, list of warnings)
    """
    warnings = []
    df = df.copy()
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        warnings.append(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
        df = df.dropna(subset=required_cols)
    
    # Validate price relationships
    invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
    invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
    
    if invalid_high.any():
        warnings.append(f"Found {invalid_high.sum()} candles with high < open/close")
        # Fix by setting high to max of open/close
        df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)
    
    if invalid_low.any():
        warnings.append(f"Found {invalid_low.sum()} candles with low > open/close")
        # Fix by setting low to min of open/close
        df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)
    
    # Check for negative values
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    negative_volume = df['volume'] < 0
    
    if negative_prices.any():
        warnings.append(f"Found {negative_prices.sum()} candles with negative prices")
        df = df[~negative_prices]
    
    if negative_volume.any():
        warnings.append(f"Found {negative_volume.sum()} candles with negative volume")
        df.loc[negative_volume, 'volume'] = 0
    
    # Check for outliers (Z-score > 3)
    for col in ['open', 'high', 'low', 'close']:
        zscore = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = zscore > 3
        if outliers.any():
            warnings.append(f"Found {outliers.sum()} outliers in {col}")
    
    # Check for gaps in timestamp index
    if isinstance(df.index, pd.DatetimeIndex):
        gaps = check_timestamp_gaps(df)
        if gaps:
            warnings.append(f"Found {len(gaps)} gaps in time series")
    
    return df, warnings

def check_timestamp_gaps(df: pd.DataFrame, 
                        max_gap: pd.Timedelta = pd.Timedelta(minutes=5)) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Check for gaps in timestamp series
    
    Args:
        df: DataFrame with DatetimeIndex
        max_gap: Maximum allowed gap between timestamps
        
    Returns:
        List of (start, end) timestamps where gaps occur
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
        
    gaps = []
    diff = df.index[1:] - df.index[:-1]
    large_gaps = diff > max_gap
    
    if large_gaps.any():
        gap_starts = df.index[:-1][large_gaps]
        gap_ends = df.index[1:][large_gaps]
        gaps = list(zip(gap_starts, gap_ends))
        
    return gaps

def resample_ohlcv(df: pd.DataFrame, 
                   freq: str = '5min',
                   fill_gaps: bool = True) -> pd.DataFrame:
    """
    Resample OHLCV data to a fixed frequency
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        freq: Target frequency (e.g., '1min', '5min', '15min')
        fill_gaps: Whether to forward fill missing values
        
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Resample rules for OHLCV
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(freq).first()
    resampled['high'] = df['high'].resample(freq).max()
    resampled['low'] = df['low'].resample(freq).min()
    resampled['close'] = df['close'].resample(freq).last()
    resampled['volume'] = df['volume'].resample(freq).sum()
    
    if fill_gaps:
        # Forward fill prices, fill volume with 0
        resampled[['open', 'high', 'low', 'close']] = resampled[['open', 'high', 'low', 'close']].ffill()
        resampled['volume'] = resampled['volume'].fillna(0)
    
    return resampled