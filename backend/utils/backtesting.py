from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    positions: pd.DataFrame

class Backtester:
    def __init__(self, 
                initial_capital: float = 100000,
                commission: float = 0.001,  # 0.1% commission per trade
                slippage: float = 0.0002):  # 0.02% slippage
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run(self, 
           data: pd.DataFrame, 
           signals: List[Dict],
           position_sizes: Dict[str, float] = None) -> BacktestResult:
        """
        Run backtest on historical data with generated signals
        
        Args:
            data: OHLCV DataFrame
            signals: List of signal dictionaries with fields:
                    - timestamp
                    - security_id
                    - side (BUY/SELL)
                    - confidence
                    - quantity
            position_sizes: Dict of max position sizes by security_id
            
        Returns:
            BacktestResult object with equity curve and metrics
        """
        positions = pd.DataFrame(columns=['security_id', 'quantity', 'entry_price', 'entry_time'])
        trades = []
        equity = [self.initial_capital]
        current_capital = self.initial_capital
        
        for signal in signals:
            try:
                timestamp = signal['timestamp']
                security_id = signal['security_id']
                price = data.loc[timestamp, 'close']
                
                # Apply slippage
                exec_price = price * (1 + self.slippage) if signal['side'] == 'BUY' else price * (1 - self.slippage)
                
                # Position sizing
                if position_sizes and security_id in position_sizes:
                    max_size = position_sizes[security_id]
                    signal['quantity'] = min(signal['quantity'], max_size)
                
                # Execute trade
                trade_value = exec_price * signal['quantity']
                commission = trade_value * self.commission
                
                if signal['side'] == 'BUY':
                    # Check if we have enough capital
                    if trade_value + commission > current_capital:
                        logger.warning(f"Insufficient capital for trade at {timestamp}")
                        continue
                        
                    # Add new position
                    positions = positions.append({
                        'security_id': security_id,
                        'quantity': signal['quantity'],
                        'entry_price': exec_price,
                        'entry_time': timestamp
                    }, ignore_index=True)
                    
                    current_capital -= (trade_value + commission)
                    
                else:  # SELL
                    # Find matching position
                    pos_idx = positions[positions['security_id'] == security_id].index
                    if len(pos_idx) == 0:
                        continue
                        
                    pos = positions.loc[pos_idx[0]]
                    exit_value = pos['quantity'] * exec_price
                    pnl = exit_value - (pos['quantity'] * pos['entry_price'])
                    pnl -= (exit_value + pos['quantity'] * pos['entry_price']) * self.commission
                    
                    trades.append({
                        'security_id': security_id,
                        'entry_time': pos['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': pos['entry_price'],
                        'exit_price': exec_price,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'return': pnl / (pos['quantity'] * pos['entry_price'])
                    })
                    
                    positions = positions.drop(pos_idx[0])
                    current_capital += exit_value - commission
                
                equity.append(current_capital + self._mark_to_market(positions, data, timestamp))
                
            except Exception as e:
                logger.error(f"Error processing signal at {timestamp}: {e}")
                continue
        
        equity_curve = pd.Series(equity, name='equity')
        trades_df = pd.DataFrame(trades)
        
        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            metrics=self._calculate_metrics(equity_curve, trades_df),
            positions=positions
        )
    
    def _mark_to_market(self, positions: pd.DataFrame, data: pd.DataFrame, timestamp: pd.Timestamp) -> float:
        """Calculate current value of open positions"""
        if positions.empty:
            return 0
            
        total_value = 0
        current_price = data.loc[timestamp, 'close']
        
        for _, pos in positions.iterrows():
            total_value += pos['quantity'] * current_price
            
        return total_value
    
    def _calculate_metrics(self, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics"""
        returns = equity.pct_change().dropna()
        
        if trades.empty:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Drawdown
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        winning_trades = trades[trades['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }