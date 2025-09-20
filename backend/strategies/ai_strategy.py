from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
from datetime import datetime
from loguru import logger

from utils.feature_engineering import calculate_features, create_target
from utils.backtesting import Backtester, BacktestResult

class AITradingStrategy:
    """AI-based trading strategy using LightGBM and portfolio-level risk management"""
    
    def __init__(self, 
                model_path: str = None,
                risk_per_trade: float = 0.02,  # 2% risk per trade
                max_position_value: float = 0.1,  # Max 10% of capital per position
                confidence_threshold: float = 0.7):
        self.model = self._load_model(model_path) if model_path else None
        self.scaler = StandardScaler()
        self.risk_per_trade = risk_per_trade
        self.max_position_value = max_position_value
        self.confidence_threshold = confidence_threshold
        
        # Feature columns used by the model
        self.feature_cols = [
            'rsi_zscore', 'macd_zscore', 'macd_hist_zscore',
            'atr_pct', 'adx_zscore', 'obv_zscore',
            'vwap_diff_zscore', 'relative_volume_zscore',
            'volatility_zscore', 'returns_zscore',
            'time_of_day'
        ]
        
    def _load_model(self, model_path: str) -> Optional[lgb.Booster]:
        """Load trained model and scaler from disk"""
        try:
            if model_path:
                self.model = joblib.load(f"{model_path}_model.joblib")
                self.scaler = joblib.load(f"{model_path}_scaler.joblib")
                return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
            
    def train_model(self, historical_data: pd.DataFrame, 
                   cv_splits: int = 5,
                   early_stopping_rounds: int = 50):
        """
        Train LightGBM model with time-series cross-validation
        
        Args:
            historical_data: DataFrame with OHLCV data
            cv_splits: Number of time-series CV folds
            early_stopping_rounds: Stop training if no improvement
        """
        try:
            # Prepare features
            features = calculate_features(historical_data)
            target = create_target(historical_data)
            
            # Remove samples where target is NaN (holding periods)
            valid_idx = ~target.isna()
            features = features[valid_idx]
            target = target[valid_idx]
            
            # Select and scale features
            X = features[self.feature_cols]
            X_scaled = self.scaler.fit_transform(X)
            y = target
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            models = []
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train = X_scaled[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X_scaled[val_idx]
                y_val = y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                # LightGBM parameters
                params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'num_leaves': 32,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
                
                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False
                )
                
                # Evaluate
                val_pred = model.predict(X_val)
                score = roc_auc_score(y_val, val_pred)
                scores.append(score)
                models.append(model)
                
                logger.info(f"Fold {fold + 1} AUC: {score:.4f}")
            
            # Use best model
            best_model_idx = np.argmax(scores)
            self.model = models[best_model_idx]
            logger.info(f"Model training completed. Best AUC: {scores[best_model_idx]:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
    def save_model(self, path: str):
        """Save trained model and scaler to disk"""
        if self.model and self.scaler:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{path}_model_{timestamp}.joblib"
            scaler_path = f"{path}_scaler_{timestamp}.joblib"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            
    def backtest(self,
                data: Dict[str, pd.DataFrame],
                initial_capital: float = 100000,
                commission: float = 0.001,
                slippage: float = 0.0002) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: Dict of security_id to OHLCV DataFrame
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Estimated slippage per trade
            
        Returns:
            BacktestResult with equity curve and performance metrics
        """
        backtester = Backtester(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        # Track account state for position sizing
        account_value = initial_capital
        current_positions = {}
        all_signals = []
        
        # Generate signals for each timepoint
        for timestamp in sorted(next(iter(data.values())).index):
            # Update current data view
            current_data = {
                sid: df[df.index <= timestamp]
                for sid, df in data.items()
            }
            
            # Generate signals
            signals = self.generate_signals(
                market_data=current_data,
                account_value=account_value,
                current_positions=current_positions
            )
            
            # Add timestamp to signals
            for signal in signals:
                signal['timestamp'] = timestamp
                all_signals.append(signal)
        
        # Run backtest
        results = backtester.run(
            data=next(iter(data.values())),  # Use first security's data for timestamps
            signals=all_signals
        )
        
        logger.info("Backtest completed")
        logger.info(f"Total Return: {results.metrics['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results.metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results.metrics['max_drawdown']:.2%}")
        logger.info(f"Win Rate: {results.metrics['win_rate']:.2%}")
        
        return results
            
    def generate_signals(self, 
                        market_data: Dict[str, pd.DataFrame],
                        account_value: float,
                        current_positions: Dict[str, Dict] = None) -> List[Dict]:
        """
        Generate trading signals with portfolio-level risk management
        
        Args:
            market_data: Dict of security_id to OHLCV DataFrame
            account_value: Current total account value
            current_positions: Dict of security_id to position info
        """
        signals = []
        current_positions = current_positions or {}
        
        # Calculate available capital
        invested_value = sum(pos['value'] for pos in current_positions.values())
        available_capital = account_value - invested_value
        
        for security_id, data in market_data.items():
            try:
                # Skip if not enough data
                if len(data) < 20:  # Need enough data for features
                    continue
                    
                # Calculate features
                features = calculate_features(data)
                if features.empty or self.model is None:
                    continue
                
                # Scale features
                X = features[self.feature_cols].iloc[-1:]
                X_scaled = self.scaler.transform(X)
                
                # Get model prediction and confidence
                pred_prob = self.model.predict(X_scaled)[0]
                
                # Current position in this security
                current_pos = current_positions.get(security_id, {})
                current_qty = current_pos.get('quantity', 0)
                
                # Generate signal based on prediction and current position
                if pred_prob > self.confidence_threshold and current_qty <= 0:
                    # Strong buy signal and no long position
                    position_size = self._calculate_position_size(
                        data=data,
                        confidence=pred_prob,
                        account_value=account_value,
                        available_capital=available_capital
                    )
                    
                    if position_size > 0:
                        signals.append({
                            "timestamp": data.index[-1],
                            "security_id": security_id,
                            "side": "BUY",
                            "confidence": pred_prob,
                            "quantity": position_size,
                            "reason": self._get_signal_reason(features.iloc[-1], pred_prob)
                        })
                        
                elif pred_prob < (1 - self.confidence_threshold) and current_qty > 0:
                    # Strong sell signal and have long position
                    signals.append({
                        "timestamp": data.index[-1],
                        "security_id": security_id,
                        "side": "SELL",
                        "confidence": 1 - pred_prob,
                        "quantity": current_qty,  # Sell entire position
                        "reason": self._get_signal_reason(features.iloc[-1], pred_prob)
                    })
                    
            except Exception as e:
                logger.error(f"Error generating signal for {security_id}: {e}")
                continue
                
        return signals
        
    def _calculate_position_size(self,
                              data: pd.DataFrame,
                              confidence: float,
                              account_value: float,
                              available_capital: float) -> int:
        """
        Calculate position size based on:
        - Account risk tolerance
        - Position volatility (ATR)
        - Model confidence
        - Available capital
        """
        try:
            current_price = data['close'].iloc[-1]
            atr = data['atr'].iloc[-1]
            
            # Risk-based position size
            risk_amount = account_value * self.risk_per_trade * confidence
            price_risk = atr * 2  # Use 2 ATR as initial stop distance
            position_size = risk_amount / price_risk
            
            # Value-based limits
            max_value = account_value * self.max_position_value
            value_limit = max_value / current_price
            capital_limit = available_capital / current_price
            
            # Apply limits
            final_size = min(position_size, value_limit, capital_limit)
            
            # Round to nearest lot size (adjust as needed)
            lot_size = 1
            final_size = int(final_size / lot_size) * lot_size
            
            return max(0, final_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
            
    def _get_signal_reason(self, features: pd.Series, confidence: float) -> str:
        """Generate human-readable explanation for the signal"""
        reasons = []
        
        if features['adx_zscore'] > 1:
            reasons.append("Strong trend")
        if features['rsi_zscore'] < -2:
            reasons.append("Oversold")
        elif features['rsi_zscore'] > 2:
            reasons.append("Overbought")
        if features['relative_volume_zscore'] > 1:
            reasons.append("High volume")
        if abs(features['vwap_diff_zscore']) > 2:
            reasons.append("Significant VWAP deviation")
            
        confidence_pct = f"{confidence:.1%}"
        if reasons:
            return f"Model confidence {confidence_pct} - {', '.join(reasons)}"
        return f"Model confidence {confidence_pct}"