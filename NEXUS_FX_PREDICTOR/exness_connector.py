"""
NEXUS FX Predictor - Exness Data Connector
Data connection module for live Forex data (Simulated for demo)
"""

import pandas as pd
import numpy as np
import datetime
import json
import time
import random
from typing import Dict, List, Optional

class ExnessConnector:
    """
    Exness broker data connector (Simulated)
    In production, this would connect to real Exness API
    """
    
    def __init__(self, config_file: str = "exness_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.connected = False
        self.available_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDCAD', 'GBPAUD'
        ]
        self.current_prices = {}
        self.last_update = None
        
        # Base prices for simulation (realistic starting points)
        self.base_prices = {
            'EURUSD': 1.0950,
            'GBPUSD': 1.2680,
            'USDJPY': 149.50,
            'USDCHF': 0.8890,
            'AUDUSD': 0.6580,
            'USDCAD': 1.3720,
            'NZDUSD': 0.5940,
            'EURJPY': 163.80,
            'GBPJPY': 189.60,
            'EURGBP': 0.8640,
            'AUDCAD': 0.9030,
            'GBPAUD': 1.9260
        }
        
        # Initialize current prices
        for pair in self.available_pairs:
            self.current_prices[pair] = self.base_prices[pair]
    
    def load_config(self) -> Dict:
        """
        Load configuration from file
        """
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default config
            default_config = {
                "broker": "Exness",
                "login": "demo_account",
                "password": "demo_password",
                "server": "ExnessEU-Demo",
                "selected_pair": "EURUSD",
                "auto_login": True,
                "timeframe": "M1"
            }
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict):
        """
        Save configuration to file
        """
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def connect(self) -> bool:
        """
        Connect to Exness broker (Simulated)
        """
        try:
            # Simulate connection process
            print("Connecting to Exness...")
            time.sleep(1)
            
            # In real implementation, this would authenticate with Exness API
            self.connected = True
            print(f"Connected to Exness - Account: {self.config['login']}")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from broker
        """
        self.connected = False
        print("Disconnected from Exness")
    
    def get_available_pairs(self) -> List[str]:
        """
        Get list of available currency pairs
        """
        return self.available_pairs
    
    def set_pair(self, pair: str) -> bool:
        """
        Set the active trading pair
        """
        if pair in self.available_pairs:
            self.config['selected_pair'] = pair
            self.save_config(self.config)
            return True
        return False
    
    def get_current_pair(self) -> str:
        """
        Get the currently selected pair
        """
        return self.config.get('selected_pair', 'EURUSD')
    
    def simulate_price_movement(self, pair: str, current_price: float) -> float:
        """
        Simulate realistic price movement for demo purposes
        """
        # Calculate realistic volatility based on pair
        volatility_map = {
            'EURUSD': 0.0001, 'GBPUSD': 0.0002, 'USDJPY': 0.02,
            'USDCHF': 0.0001, 'AUDUSD': 0.0002, 'USDCAD': 0.0001,
            'NZDUSD': 0.0002, 'EURJPY': 0.03, 'GBPJPY': 0.04,
            'EURGBP': 0.0001, 'AUDCAD': 0.0002, 'GBPAUD': 0.0003
        }
        
        volatility = volatility_map.get(pair, 0.0001)
        
        # Generate realistic movement
        change = np.random.normal(0, volatility)
        
        # Add some trend bias (random walk with slight mean reversion)
        trend_factor = 0.95 + (np.random.random() * 0.1)
        new_price = current_price * trend_factor + change
        
        return round(new_price, 5 if 'JPY' not in pair else 3)
    
    def get_live_price(self, pair: str = None) -> Dict[str, float]:
        """
        Get current live price for specified pair
        """
        if not self.connected:
            if not self.connect():
                return {}
        
        if pair is None:
            pair = self.get_current_pair()
        
        if pair not in self.available_pairs:
            return {}
        
        # Simulate live price movement
        current_price = self.current_prices[pair]
        new_price = self.simulate_price_movement(pair, current_price)
        self.current_prices[pair] = new_price
        
        # Simulate bid/ask spread
        spread = 0.00015 if 'JPY' not in pair else 0.015
        bid = new_price - spread/2
        ask = new_price + spread/2
        
        return {
            'pair': pair,
            'bid': round(bid, 5 if 'JPY' not in pair else 3),
            'ask': round(ask, 5 if 'JPY' not in pair else 3),
            'mid': new_price,
            'spread': spread,
            'timestamp': datetime.datetime.now()
        }
    
    def generate_historical_data(self, pair: str, periods: int = 100) -> pd.DataFrame:
        """
        Generate historical OHLCV data for simulation
        """
        if pair not in self.available_pairs:
            return pd.DataFrame()
        
        base_price = self.base_prices[pair]
        timestamps = []
        prices = []
        
        # Generate timestamps (1-minute intervals)
        start_time = datetime.datetime.now() - datetime.timedelta(minutes=periods)
        for i in range(periods):
            timestamps.append(start_time + datetime.timedelta(minutes=i))
        
        # Generate price series with realistic movement
        current_price = base_price
        for i in range(periods):
            current_price = self.simulate_price_movement(pair, current_price)
            prices.append(current_price)
        
        # Generate OHLCV data
        ohlcv_data = []
        for i in range(len(timestamps)):
            if i == 0:
                open_price = base_price
            else:
                open_price = ohlcv_data[-1]['close']
            
            close_price = prices[i]
            
            # Generate realistic high/low based on open/close
            mid_price = (open_price + close_price) / 2
            volatility_map = {
                'EURUSD': 0.0005, 'GBPUSD': 0.0008, 'USDJPY': 0.05,
                'USDCHF': 0.0005, 'AUDUSD': 0.0008, 'USDCAD': 0.0005,
                'NZDUSD': 0.0008, 'EURJPY': 0.08, 'GBPJPY': 0.12,
                'EURGBP': 0.0005, 'AUDCAD': 0.0008, 'GBPAUD': 0.0012
            }
            
            volatility = volatility_map.get(pair, 0.0005)
            range_size = np.random.uniform(volatility/2, volatility * 2)
            
            high_price = max(open_price, close_price) + np.random.uniform(0, range_size)
            low_price = min(open_price, close_price) - np.random.uniform(0, range_size)
            
            # Ensure high is actually highest and low is lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate realistic volume (forex doesn't have true volume, so we simulate tick volume)
            volume = np.random.randint(50, 500)
            
            ohlcv_data.append({
                'timestamp': timestamps[i],
                'open': round(open_price, 5 if 'JPY' not in pair else 3),
                'high': round(high_price, 5 if 'JPY' not in pair else 3),
                'low': round(low_price, 5 if 'JPY' not in pair else 3),
                'close': round(close_price, 5 if 'JPY' not in pair else 3),
                'volume': volume
            })
        
        df = pd.DataFrame(ohlcv_data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_historical_data(self, pair: str = None, periods: int = 100) -> pd.DataFrame:
        """
        Get historical OHLCV data for specified pair
        """
        if not self.connected:
            if not self.connect():
                return pd.DataFrame()
        
        if pair is None:
            pair = self.get_current_pair()
        
        # In real implementation, this would fetch from Exness API
        # For demo, we generate realistic data
        return self.generate_historical_data(pair, periods)
    
    def get_live_ohlcv_data(self, pair: str = None, periods: int = 100) -> pd.DataFrame:
        """
        Get live OHLCV data including most recent candle
        """
        # Get historical data
        historical_data = self.get_historical_data(pair, periods - 1)
        
        if historical_data.empty:
            return pd.DataFrame()
        
        # Add current live candle
        live_price_data = self.get_live_price(pair)
        if live_price_data:
            current_time = datetime.datetime.now().replace(second=0, microsecond=0)
            last_close = historical_data.iloc[-1]['close']
            
            # Create current minute candle
            current_candle = {
                'open': last_close,
                'high': max(last_close, live_price_data['mid']),
                'low': min(last_close, live_price_data['mid']),
                'close': live_price_data['mid'],
                'volume': np.random.randint(50, 200)
            }
            
            # Add to dataframe
            new_row = pd.DataFrame([current_candle], index=[current_time])
            historical_data = pd.concat([historical_data, new_row])
        
        return historical_data
    
    def test_connection(self) -> Dict[str, any]:
        """
        Test broker connection and return status
        """
        if not self.connected:
            self.connect()
        
        status = {
            'connected': self.connected,
            'broker': self.config.get('broker', 'Unknown'),
            'account': self.config.get('login', 'Unknown'),
            'server': self.config.get('server', 'Unknown'),
            'available_pairs': len(self.available_pairs),
            'current_pair': self.get_current_pair(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if self.connected:
            # Test by getting a live price
            test_price = self.get_live_price()
            status['test_price'] = test_price
            status['status'] = 'OK'
        else:
            status['status'] = 'FAILED'
        
        return status

# Create configuration file template
def create_default_config():
    """
    Create default configuration file
    """
    config = {
        "broker": "Exness",
        "login": "your_demo_account",
        "password": "your_password",
        "server": "ExnessEU-Demo",
        "selected_pair": "EURUSD",
        "auto_login": True,
        "timeframe": "M1",
        "notes": "Replace with your actual Exness demo account credentials"
    }
    
    with open("NEXUS_FX_PREDICTOR/exness_config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Default configuration created: exness_config.json")
    print("Please update with your actual Exness credentials")

if __name__ == "__main__":
    # Test the connector
    connector = ExnessConnector()
    
    print("Testing Exness Connector...")
    status = connector.test_connection()
    print(json.dumps(status, indent=2, default=str))
    
    print("\nGetting historical data...")
    data = connector.get_historical_data('EURUSD', 20)
    print(data.tail())
    
    print("\nAvailable pairs:")
    print(connector.get_available_pairs())