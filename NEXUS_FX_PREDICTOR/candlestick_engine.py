"""
NEXUS FX Predictor - Candlestick Pattern Recognition Engine
Advanced candlestick pattern detection for Forex analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class CandlestickPatterns:
    """
    Advanced candlestick pattern recognition engine
    """
    
    def __init__(self):
        self.patterns = {}
        self.min_body_ratio = 0.1  # Minimum body size relative to full range
        self.doji_threshold = 0.05  # Maximum body size for doji pattern
        
    @staticmethod
    def get_candle_properties(open_price: float, high_price: float, low_price: float, close_price: float) -> Dict[str, float]:
        """
        Calculate basic candle properties
        """
        body = abs(close_price - open_price)
        full_range = high_price - low_price
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        body_ratio = body / full_range if full_range > 0 else 0
        upper_shadow_ratio = upper_shadow / full_range if full_range > 0 else 0
        lower_shadow_ratio = lower_shadow / full_range if full_range > 0 else 0
        
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        is_doji = body_ratio < 0.05
        
        return {
            'body': body,
            'full_range': full_range,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'is_bullish': is_bullish,
            'is_bearish': is_bearish,
            'is_doji': is_doji
        }
    
    def detect_doji(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Doji pattern - indecision candle
        """
        props = self.get_candle_properties(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
        return props['body_ratio'] < self.doji_threshold
    
    def detect_hammer(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Hammer pattern - bullish reversal
        """
        props = self.get_candle_properties(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
        
        return (props['lower_shadow_ratio'] >= 0.6 and  # Long lower shadow
                props['upper_shadow_ratio'] <= 0.1 and  # Short upper shadow
                props['body_ratio'] >= 0.1)  # Decent body size
    
    def detect_hanging_man(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Hanging Man pattern - bearish reversal (same shape as hammer)
        """
        return self.detect_hammer(ohlc)  # Same pattern, context determines meaning
    
    def detect_shooting_star(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Shooting Star pattern - bearish reversal
        """
        props = self.get_candle_properties(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
        
        return (props['upper_shadow_ratio'] >= 0.6 and  # Long upper shadow
                props['lower_shadow_ratio'] <= 0.1 and  # Short lower shadow
                props['body_ratio'] >= 0.1)  # Decent body size
    
    def detect_inverted_hammer(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Inverted Hammer pattern - bullish reversal (same shape as shooting star)
        """
        return self.detect_shooting_star(ohlc)  # Same pattern, context determines meaning
    
    def detect_bullish_engulfing(self, prev_ohlc: Dict[str, float], curr_ohlc: Dict[str, float]) -> bool:
        """
        Detect Bullish Engulfing pattern
        """
        prev_props = self.get_candle_properties(prev_ohlc['open'], prev_ohlc['high'], prev_ohlc['low'], prev_ohlc['close'])
        curr_props = self.get_candle_properties(curr_ohlc['open'], curr_ohlc['high'], curr_ohlc['low'], curr_ohlc['close'])
        
        return (prev_props['is_bearish'] and  # Previous candle is bearish
                curr_props['is_bullish'] and  # Current candle is bullish
                curr_ohlc['open'] < prev_ohlc['close'] and  # Current opens below previous close
                curr_ohlc['close'] > prev_ohlc['open'] and  # Current closes above previous open
                curr_props['body'] > prev_props['body'])  # Current body is larger
    
    def detect_bearish_engulfing(self, prev_ohlc: Dict[str, float], curr_ohlc: Dict[str, float]) -> bool:
        """
        Detect Bearish Engulfing pattern
        """
        prev_props = self.get_candle_properties(prev_ohlc['open'], prev_ohlc['high'], prev_ohlc['low'], prev_ohlc['close'])
        curr_props = self.get_candle_properties(curr_ohlc['open'], curr_ohlc['high'], curr_ohlc['low'], curr_ohlc['close'])
        
        return (prev_props['is_bullish'] and  # Previous candle is bullish
                curr_props['is_bearish'] and  # Current candle is bearish
                curr_ohlc['open'] > prev_ohlc['close'] and  # Current opens above previous close
                curr_ohlc['close'] < prev_ohlc['open'] and  # Current closes below previous open
                curr_props['body'] > prev_props['body'])  # Current body is larger
    
    def detect_dark_cloud_cover(self, prev_ohlc: Dict[str, float], curr_ohlc: Dict[str, float]) -> bool:
        """
        Detect Dark Cloud Cover pattern - bearish reversal
        """
        prev_props = self.get_candle_properties(prev_ohlc['open'], prev_ohlc['high'], prev_ohlc['low'], prev_ohlc['close'])
        curr_props = self.get_candle_properties(curr_ohlc['open'], curr_ohlc['high'], curr_ohlc['low'], curr_ohlc['close'])
        
        return (prev_props['is_bullish'] and  # Previous candle is bullish
                curr_props['is_bearish'] and  # Current candle is bearish
                curr_ohlc['open'] > prev_ohlc['high'] and  # Gap up opening
                curr_ohlc['close'] < (prev_ohlc['open'] + prev_ohlc['close']) / 2)  # Closes below midpoint
    
    def detect_piercing_pattern(self, prev_ohlc: Dict[str, float], curr_ohlc: Dict[str, float]) -> bool:
        """
        Detect Piercing Pattern - bullish reversal
        """
        prev_props = self.get_candle_properties(prev_ohlc['open'], prev_ohlc['high'], prev_ohlc['low'], prev_ohlc['close'])
        curr_props = self.get_candle_properties(curr_ohlc['open'], curr_ohlc['high'], curr_ohlc['low'], curr_ohlc['close'])
        
        return (prev_props['is_bearish'] and  # Previous candle is bearish
                curr_props['is_bullish'] and  # Current candle is bullish
                curr_ohlc['open'] < prev_ohlc['low'] and  # Gap down opening
                curr_ohlc['close'] > (prev_ohlc['open'] + prev_ohlc['close']) / 2)  # Closes above midpoint
    
    def detect_morning_star(self, first_ohlc: Dict[str, float], second_ohlc: Dict[str, float], third_ohlc: Dict[str, float]) -> bool:
        """
        Detect Morning Star pattern - bullish reversal (3-candle pattern)
        """
        first_props = self.get_candle_properties(first_ohlc['open'], first_ohlc['high'], first_ohlc['low'], first_ohlc['close'])
        second_props = self.get_candle_properties(second_ohlc['open'], second_ohlc['high'], second_ohlc['low'], second_ohlc['close'])
        third_props = self.get_candle_properties(third_ohlc['open'], third_ohlc['high'], third_ohlc['low'], third_ohlc['close'])
        
        return (first_props['is_bearish'] and  # First candle is bearish
                second_props['body_ratio'] < 0.3 and  # Second candle has small body
                third_props['is_bullish'] and  # Third candle is bullish
                second_ohlc['high'] < first_ohlc['close'] and  # Gap down
                third_ohlc['open'] > second_ohlc['high'] and  # Gap up
                third_ohlc['close'] > (first_ohlc['open'] + first_ohlc['close']) / 2)  # Third closes above first's midpoint
    
    def detect_evening_star(self, first_ohlc: Dict[str, float], second_ohlc: Dict[str, float], third_ohlc: Dict[str, float]) -> bool:
        """
        Detect Evening Star pattern - bearish reversal (3-candle pattern)
        """
        first_props = self.get_candle_properties(first_ohlc['open'], first_ohlc['high'], first_ohlc['low'], first_ohlc['close'])
        second_props = self.get_candle_properties(second_ohlc['open'], second_ohlc['high'], second_ohlc['low'], second_ohlc['close'])
        third_props = self.get_candle_properties(third_ohlc['open'], third_ohlc['high'], third_ohlc['low'], third_ohlc['close'])
        
        return (first_props['is_bullish'] and  # First candle is bullish
                second_props['body_ratio'] < 0.3 and  # Second candle has small body
                third_props['is_bearish'] and  # Third candle is bearish
                second_ohlc['low'] > first_ohlc['close'] and  # Gap up
                third_ohlc['open'] < second_ohlc['low'] and  # Gap down
                third_ohlc['close'] < (first_ohlc['open'] + first_ohlc['close']) / 2)  # Third closes below first's midpoint
    
    def detect_tweezer_tops(self, prev_ohlc: Dict[str, float], curr_ohlc: Dict[str, float]) -> bool:
        """
        Detect Tweezer Tops pattern - bearish reversal
        """
        tolerance = abs(prev_ohlc['high'] - curr_ohlc['high']) / prev_ohlc['high']
        
        return (tolerance < 0.001 and  # Similar highs
                prev_ohlc['close'] > prev_ohlc['open'] and  # First candle bullish
                curr_ohlc['close'] < curr_ohlc['open'])  # Second candle bearish
    
    def detect_tweezer_bottoms(self, prev_ohlc: Dict[str, float], curr_ohlc: Dict[str, float]) -> bool:
        """
        Detect Tweezer Bottoms pattern - bullish reversal
        """
        tolerance = abs(prev_ohlc['low'] - curr_ohlc['low']) / prev_ohlc['low']
        
        return (tolerance < 0.001 and  # Similar lows
                prev_ohlc['close'] < prev_ohlc['open'] and  # First candle bearish
                curr_ohlc['close'] > curr_ohlc['open'])  # Second candle bullish
    
    def detect_marubozu_bullish(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Bullish Marubozu - strong bullish candle with no shadows
        """
        props = self.get_candle_properties(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
        
        return (props['is_bullish'] and
                props['upper_shadow_ratio'] < 0.02 and
                props['lower_shadow_ratio'] < 0.02 and
                props['body_ratio'] > 0.95)
    
    def detect_marubozu_bearish(self, ohlc: Dict[str, float]) -> bool:
        """
        Detect Bearish Marubozu - strong bearish candle with no shadows
        """
        props = self.get_candle_properties(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])
        
        return (props['is_bearish'] and
                props['upper_shadow_ratio'] < 0.02 and
                props['lower_shadow_ratio'] < 0.02 and
                props['body_ratio'] > 0.95)
    
    def analyze_patterns(self, ohlcv_data: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze all candlestick patterns for the given data
        """
        if len(ohlcv_data) < 3:
            return {'patterns': [], 'latest_patterns': [], 'signal': 'NEUTRAL'}
        
        patterns_found = []
        latest_patterns = []
        
        # Convert to list of dictionaries for easier processing
        candles = []
        for idx, row in ohlcv_data.iterrows():
            candles.append({
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'timestamp': row.get('timestamp', idx)
            })
        
        # Analyze patterns
        for i in range(len(candles)):
            current_patterns = []
            
            # Single candle patterns
            if self.detect_doji(candles[i]):
                current_patterns.append('Doji')
            if self.detect_hammer(candles[i]):
                current_patterns.append('Hammer')
            if self.detect_shooting_star(candles[i]):
                current_patterns.append('Shooting Star')
            if self.detect_marubozu_bullish(candles[i]):
                current_patterns.append('Bullish Marubozu')
            if self.detect_marubozu_bearish(candles[i]):
                current_patterns.append('Bearish Marubozu')
            
            # Two candle patterns
            if i > 0:
                if self.detect_bullish_engulfing(candles[i-1], candles[i]):
                    current_patterns.append('Bullish Engulfing')
                if self.detect_bearish_engulfing(candles[i-1], candles[i]):
                    current_patterns.append('Bearish Engulfing')
                if self.detect_dark_cloud_cover(candles[i-1], candles[i]):
                    current_patterns.append('Dark Cloud Cover')
                if self.detect_piercing_pattern(candles[i-1], candles[i]):
                    current_patterns.append('Piercing Pattern')
                if self.detect_tweezer_tops(candles[i-1], candles[i]):
                    current_patterns.append('Tweezer Tops')
                if self.detect_tweezer_bottoms(candles[i-1], candles[i]):
                    current_patterns.append('Tweezer Bottoms')
            
            # Three candle patterns
            if i > 1:
                if self.detect_morning_star(candles[i-2], candles[i-1], candles[i]):
                    current_patterns.append('Morning Star')
                if self.detect_evening_star(candles[i-2], candles[i-1], candles[i]):
                    current_patterns.append('Evening Star')
            
            if current_patterns:
                patterns_found.extend(current_patterns)
                if i >= len(candles) - 3:  # Last 3 candles
                    latest_patterns.extend(current_patterns)
        
        # Determine overall signal based on latest patterns
        bullish_patterns = ['Hammer', 'Bullish Engulfing', 'Piercing Pattern', 'Morning Star', 
                           'Tweezer Bottoms', 'Bullish Marubozu']
        bearish_patterns = ['Shooting Star', 'Bearish Engulfing', 'Dark Cloud Cover', 'Evening Star',
                           'Tweezer Tops', 'Bearish Marubozu']
        
        bullish_count = sum(1 for pattern in latest_patterns if pattern in bullish_patterns)
        bearish_count = sum(1 for pattern in latest_patterns if pattern in bearish_patterns)
        
        if bullish_count > bearish_count:
            signal = 'BULLISH'
        elif bearish_count > bullish_count:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'patterns': list(set(patterns_found)),
            'latest_patterns': list(set(latest_patterns)),
            'signal': signal,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'pattern_strength': max(bullish_count, bearish_count)
        }
    
    def get_pattern_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all patterns
        """
        return {
            'Doji': 'Indecision - Equal open/close prices',
            'Hammer': 'Bullish reversal - Long lower shadow',
            'Shooting Star': 'Bearish reversal - Long upper shadow',
            'Bullish Engulfing': 'Strong bullish - Current candle engulfs previous',
            'Bearish Engulfing': 'Strong bearish - Current candle engulfs previous',
            'Dark Cloud Cover': 'Bearish reversal - Opens above, closes below midpoint',
            'Piercing Pattern': 'Bullish reversal - Opens below, closes above midpoint',
            'Morning Star': 'Bullish reversal - 3-candle pattern',
            'Evening Star': 'Bearish reversal - 3-candle pattern',
            'Tweezer Tops': 'Bearish reversal - Equal highs',
            'Tweezer Bottoms': 'Bullish reversal - Equal lows',
            'Bullish Marubozu': 'Strong bullish - No shadows',
            'Bearish Marubozu': 'Strong bearish - No shadows'
        }