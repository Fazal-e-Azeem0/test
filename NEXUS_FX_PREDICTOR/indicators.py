"""
NEXUS FX Predictor - Technical Indicators Engine
Advanced technical analysis indicators for Forex prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class TechnicalIndicators:
    """
    Advanced technical indicators calculator for Forex analysis
    """
    
    @staticmethod
    def calculate_rsi(close_prices: np.array, period: int = 14) -> np.array:
        """
        Calculate Relative Strength Index (RSI)
        """
        if len(close_prices) < period + 1:
            return np.array([50.0] * len(close_prices))
        
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # Initial values
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Calculate RSI using Wilder's smoothing
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Fill initial values
        rsi[:period] = 50.0
        
        return rsi
    
    @staticmethod
    def calculate_macd(close_prices: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.array]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        """
        if len(close_prices) < slow:
            length = len(close_prices)
            return {
                'macd': np.zeros(length),
                'signal': np.zeros(length),
                'histogram': np.zeros(length)
            }
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators.calculate_ema(close_prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(close_prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_ema(prices: np.array, period: int) -> np.array:
        """
        Calculate Exponential Moving Average (EMA)
        """
        if len(prices) == 0:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def calculate_sma(prices: np.array, period: int) -> np.array:
        """
        Calculate Simple Moving Average (SMA)
        """
        if len(prices) < period:
            return np.full(len(prices), np.mean(prices))
        
        sma = np.zeros_like(prices)
        sma[:period-1] = np.mean(prices[:period])
        
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        
        return sma
    
    @staticmethod
    def calculate_bollinger_bands(close_prices: np.array, period: int = 20, std_dev: float = 2.0) -> Dict[str, np.array]:
        """
        Calculate Bollinger Bands
        """
        if len(close_prices) < period:
            middle = np.full(len(close_prices), np.mean(close_prices))
            std = np.std(close_prices)
            return {
                'upper': middle + (std_dev * std),
                'middle': middle,
                'lower': middle - (std_dev * std)
            }
        
        sma = TechnicalIndicators.calculate_sma(close_prices, period)
        std = np.zeros_like(close_prices)
        
        for i in range(period-1, len(close_prices)):
            std[i] = np.std(close_prices[i-period+1:i+1])
        
        # Fill initial values
        std[:period-1] = np.std(close_prices[:period])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_stochastic(high: np.array, low: np.array, close: np.array, k_period: int = 14, d_period: int = 3) -> Dict[str, np.array]:
        """
        Calculate Stochastic Oscillator
        """
        if len(close) < k_period:
            return {
                'k': np.full(len(close), 50.0),
                'd': np.full(len(close), 50.0)
            }
        
        k_percent = np.zeros_like(close)
        
        for i in range(k_period-1, len(close)):
            highest_high = np.max(high[i-k_period+1:i+1])
            lowest_low = np.min(low[i-k_period+1:i+1])
            
            if highest_high != lowest_low:
                k_percent[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                k_percent[i] = 50.0
        
        # Fill initial values
        k_percent[:k_period-1] = 50.0
        
        # %D is SMA of %K
        d_percent = TechnicalIndicators.calculate_sma(k_percent, d_period)
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def calculate_all_indicators(ohlcv_data: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate all technical indicators for the given OHLCV data
        """
        if ohlcv_data.empty:
            return {}
        
        close = ohlcv_data['close'].values
        high = ohlcv_data['high'].values
        low = ohlcv_data['low'].values
        
        # Calculate all indicators
        indicators = {}
        
        # RSI
        indicators['rsi'] = TechnicalIndicators.calculate_rsi(close)
        
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(close)
        indicators.update(macd_data)
        
        # EMAs
        indicators['ema_9'] = TechnicalIndicators.calculate_ema(close, 9)
        indicators['ema_21'] = TechnicalIndicators.calculate_ema(close, 21)
        indicators['ema_50'] = TechnicalIndicators.calculate_ema(close, 50)
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(close)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_middle'] = bb_data['middle']
        indicators['bb_lower'] = bb_data['lower']
        
        # Stochastic
        stoch_data = TechnicalIndicators.calculate_stochastic(high, low, close)
        indicators['stoch_k'] = stoch_data['k']
        indicators['stoch_d'] = stoch_data['d']
        
        # Get latest values for current analysis
        latest_idx = -1
        current_indicators = {
            'rsi': round(indicators['rsi'][latest_idx], 2),
            'macd': round(indicators['macd'][latest_idx], 5),
            'macd_signal': round(indicators['signal'][latest_idx], 5),
            'macd_histogram': round(indicators['histogram'][latest_idx], 5),
            'ema_9': round(indicators['ema_9'][latest_idx], 5),
            'ema_21': round(indicators['ema_21'][latest_idx], 5),
            'ema_50': round(indicators['ema_50'][latest_idx], 5),
            'bb_upper': round(indicators['bb_upper'][latest_idx], 5),
            'bb_middle': round(indicators['bb_middle'][latest_idx], 5),
            'bb_lower': round(indicators['bb_lower'][latest_idx], 5),
            'stoch_k': round(indicators['stoch_k'][latest_idx], 2),
            'stoch_d': round(indicators['stoch_d'][latest_idx], 2)
        }
        
        return {
            'all_data': indicators,
            'current': current_indicators
        }
    
    @staticmethod
    def get_signal_strength(indicators: Dict[str, float]) -> Dict[str, any]:
        """
        Analyze signal strength based on multiple indicators
        """
        signals = []
        strength = 0
        
        # RSI Analysis
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            signals.append("RSI Overbought")
            strength -= 1
        elif rsi < 30:
            signals.append("RSI Oversold")
            strength += 1
        elif 40 <= rsi <= 60:
            signals.append("RSI Neutral")
        
        # MACD Analysis
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            signals.append("MACD Bullish")
            strength += 1
        else:
            signals.append("MACD Bearish")
            strength -= 1
        
        # EMA Trend Analysis
        ema_9 = indicators.get('ema_9', 0)
        ema_21 = indicators.get('ema_21', 0)
        ema_50 = indicators.get('ema_50', 0)
        
        if ema_9 > ema_21 > ema_50:
            signals.append("Strong Uptrend")
            strength += 2
        elif ema_9 < ema_21 < ema_50:
            signals.append("Strong Downtrend")
            strength -= 2
        
        # Stochastic Analysis
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k > 80:
            signals.append("Stoch Overbought")
            strength -= 1
        elif stoch_k < 20:
            signals.append("Stoch Oversold")
            strength += 1
        
        # Determine overall signal
        if strength >= 3:
            overall_signal = "STRONG BUY"
        elif strength >= 1:
            overall_signal = "BUY"
        elif strength <= -3:
            overall_signal = "STRONG SELL"
        elif strength <= -1:
            overall_signal = "SELL"
        else:
            overall_signal = "NEUTRAL"
        
        return {
            'signal': overall_signal,
            'strength': strength,
            'individual_signals': signals,
            'confidence': min(abs(strength) * 15, 85)
        }