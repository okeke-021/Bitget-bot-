import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import logging
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BitgetFuturesBot:
    def __init__(self, config):
        """Initialize bot with configuration"""
        self.exchange = ccxt.bitget({
            'apiKey': config['api_key'],
            'secret': config['api_secret'],
            'password': config['passphrase'],
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        
        self.symbol = config['symbol']
        self.risk_percentage = config['risk_percentage']
        self.check_interval = config.get('check_interval', 30)  # seconds
        self.position = None
        self.trades_history = []
        self.total_profit = 0
        
        logger.info(f"Bot initialized for {self.symbol} with {self.risk_percentage}% risk")
    
    def fetch_ohlcv(self, timeframe='5m', limit=200):
        """Fetch OHLCV data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Trend Indicators
        df['sma_20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(df['close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Momentum Indicators
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility Indicators
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        atr = AverageTrueRange(df['high'], df['low'], df['close'])
        df['atr'] = atr.average_true_range()
        
        # Volume Indicators
        df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Price Action
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    def analyze_trend(self, df):
        """Analyze market trend"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        trend_score = 0
        
        # SMA Trend
        if latest['close'] > latest['sma_20'] > latest['sma_50']:
            trend_score += 2
        elif latest['close'] < latest['sma_20'] < latest['sma_50']:
            trend_score -= 2
            
        # EMA Crossover
        if latest['ema_12'] > latest['ema_26'] and prev['ema_12'] <= prev['ema_26']:
            trend_score += 2
        elif latest['ema_12'] < latest['ema_26'] and prev['ema_12'] >= prev['ema_26']:
            trend_score -= 2
            
        # MACD
        if latest['macd'] > latest['macd_signal'] and latest['macd_diff'] > 0:
            trend_score += 1
        elif latest['macd'] < latest['macd_signal'] and latest['macd_diff'] < 0:
            trend_score -= 1
            
        return trend_score
    
    def analyze_momentum(self, df):
        """Analyze momentum"""
        latest = df.iloc[-1]
        momentum_score = 0
        
        # RSI
        if latest['rsi'] < 30:
            momentum_score += 2  # Oversold
        elif latest['rsi'] > 70:
            momentum_score -= 2  # Overbought
        elif 40 < latest['rsi'] < 60:
            momentum_score += 0  # Neutral
            
        # Stochastic
        if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
            momentum_score += 1
        elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
            momentum_score -= 1
            
        return momentum_score
    
    def analyze_volatility(self, df):
        """Analyze volatility for risk assessment"""
        latest = df.iloc[-1]
        
        volatility_level = "LOW"
        if latest['bb_width'] > 0.04:
            volatility_level = "HIGH"
        elif latest['bb_width'] > 0.02:
            volatility_level = "MEDIUM"
            
        return volatility_level, latest['atr']
    
    def analyze_volume(self, df):
        """Analyze volume trends"""
        latest = df.iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        
        volume_score = 0
        if latest['volume'] > avg_volume * 1.5:
            volume_score = 2
        elif latest['volume'] > avg_volume:
            volume_score = 1
        else:
            volume_score = -1
            
        return volume_score
    
    def generate_signal(self, df):
        """Generate trading signal"""
        trend_score = self.analyze_trend(df)
        momentum_score = self.analyze_momentum(df)
        volume_score = self.analyze_volume(df)
        volatility_level, atr = self.analyze_volatility(df)
        
        total_score = trend_score + momentum_score + volume_score
        
        latest = df.iloc[-1]
        
        signal = {
            'action': 'HOLD',
            'confidence': 0,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'volatility': volatility_level,
            'atr': atr,
            'price': latest['close'],
            'rsi': latest['rsi']
        }
        
        # Buy signal
        if total_score >= 4 and latest['rsi'] < 65 and volume_score > 0:
            signal['action'] = 'BUY'
            signal['confidence'] = min(total_score * 10, 95)
            
        # Sell signal
        elif total_score <= -4 and latest['rsi'] > 35 and volume_score > 0:
            signal['action'] = 'SELL'
            signal['confidence'] = min(abs(total_score) * 10, 95)
            
        return signal
    
    def get_account_balance(self):
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0
    
    def calculate_position_size(self, price):
        """Calculate position size"""
        balance = self.get_account_balance()
        position_value = balance * (self.risk_percentage / 100)
        position_size = position_value / price
        return round(position_size, 4)
    
    def open_position(self, signal):
        """Open a futures position"""
        try:
            size = self.calculate_position_size(signal['price'])
            
            if signal['action'] == 'BUY':
                order = self.exchange.create_market_buy_order(self.symbol, size)
                side = 'long'
            else:
                order = self.exchange.create_market_sell_order(self.symbol, size)
                side = 'short'
            
            self.position = {
                'side': side,
                'entry_price': signal['price'],
                'size': size,
                'entry_time': datetime.now(),
                'signal': signal
            }
            
            logger.info(f"✅ Opened {side} position: {size} @ ${signal['price']:.2f}")
            logger.info(f"   Confidence: {signal['confidence']}% | RSI: {signal['rsi']:.1f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error opening position: {e}")
            return False
    
    def close_position(self, current_price, reason=""):
        """Close current position"""
        try:
            if not self.position:
                return False
            
            size = self.position['size']
            
            if self.position['side'] == 'long':
                order = self.exchange.create_market_sell_order(self.symbol, size)
                pnl = (current_price - self.position['entry_price']) * size
            else:
                order = self.exchange.create_market_buy_order(self.symbol, size)
                pnl = (self.position['entry_price'] - current_price) * size
            
            pnl_pct = (pnl / (self.position['entry_price'] * size)) * 100
            self.total_profit += pnl
            
            trade_record = {
                'side': self.position['side'],
                'entry_price': self.position['entry_price'],
                'exit_price': current_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_time': self.position['entry_time'].isoformat(),
                'exit_time': datetime.now().isoformat(),
                'reason': reason
            }
            
            self.trades_history.append(trade_record)
            self.save_trade_history()
            
            logger.info(f"🔒 Closed {self.position['side']} position")
            logger.info(f"   Entry: ${self.position['entry_price']:.2f} | Exit: ${current_price:.2f}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
            logger.info(f"   Total Profit: ${self.total_profit:.2f}")
            
            self.position = None
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return False
    
    def check_exit_conditions(self, df, current_price):
        """Check if position should be closed"""
        if not self.position:
            return False, ""
        
        entry_price = self.position['entry_price']
        side = self.position['side']
        
        # Calculate PnL percentage
        if side == 'long':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Take profit at 2%
        if pnl_pct >= 2:
            return True, f"Take Profit ({pnl_pct:.2f}%)"
        
        # Stop loss at -1%
        if pnl_pct <= -1:
            return True, f"Stop Loss ({pnl_pct:.2f}%)"
        
        # Check for reversal signals
        signal = self.generate_signal(df)
        
        if side == 'long' and signal['action'] == 'SELL' and signal['confidence'] > 60:
            return True, "Reversal Signal"
        elif side == 'short' and signal['action'] == 'BUY' and signal['confidence'] > 60:
            return True, "Reversal Signal"
        
        return False, ""
    
    def save_trade_history(self):
        """Save trade history to file"""
        try:
            with open('trade_history.json', 'w') as f:
                json.dump({
                    'trades': self.trades_history,
                    'total_profit': self.total_profit,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")
    
    def load_trade_history(self):
        """Load trade history from file"""
        try:
            if os.path.exists('trade_history.json'):
                with open('trade_history.json', 'r') as f:
                    data = json.load(f)
                    self.trades_history = data.get('trades', [])
                    self.total_profit = data.get('total_profit', 0)
                    logger.info(f"Loaded {len(self.trades_history)} trades from history")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
    
    def print_status(self, signal, current_price):
        """Print current status"""
        balance = self.get_account_balance()
        
        logger.info("=" * 70)
        logger.info(f"📊 Status Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"💰 Balance: ${balance:.2f} | Total Profit: ${self.total_profit:.2f}")
        logger.info(f"💵 Price: ${current_price:.2f} | Signal: {signal['action']} ({signal['confidence']}%)")
        logger.info(f"📈 Trend: {signal['trend_score']} | Momentum: {signal['momentum_score']} | Volume: {signal['volume_score']}")
        logger.info(f"📊 RSI: {signal['rsi']:.1f} | Volatility: {signal['volatility']}")
        
        if self.position:
            entry = self.position['entry_price']
            if self.position['side'] == 'long':
                pnl_pct = ((current_price - entry) / entry) * 100
            else:
                pnl_pct = ((entry - current_price) / entry) * 100
            
            logger.info(f"🎯 Position: {self.position['side'].upper()} | Entry: ${entry:.2f} | P&L: {pnl_pct:+.2f}%")
        else:
            logger.info("🎯 Position: None")
        
        logger.info("=" * 70)
    
    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Bitget Futures Trading Bot")
        logger.info(f"📊 Trading: {self.symbol}")
        logger.info(f"💰 Risk per trade: {self.risk_percentage}%")
        logger.info(f"⏱️  Check interval: {self.check_interval}s")
        
        # Load previous trade history
        self.load_trade_history()
        
        if self.trades_history:
            wins = len([t for t in self.trades_history if t['pnl'] > 0])
            total = len(self.trades_history)
            win_rate = (wins / total * 100) if total > 0 else 0
            logger.info(f"📈 Previous Stats: {total} trades | Win Rate: {win_rate:.1f}%")
        
        logger.info("=" * 70)
        
        try:
            while True:
                # Fetch and analyze data
                df = self.fetch_ohlcv()
                
                if df is None:
                    logger.warning("⚠️  Failed to fetch data, retrying...")
                    time.sleep(self.check_interval)
                    continue
                
                df = self.calculate_indicators(df)
                signal = self.generate_signal(df)
                current_price = df.iloc[-1]['close']
                
                # Print status
                self.print_status(signal, current_price)
                
                # Trading logic
                if self.position:
                    # Check exit conditions
                    should_close, reason = self.check_exit_conditions(df, current_price)
                    
                    if should_close:
                        self.close_position(current_price, reason)
                else:
                    # Check entry conditions
                    if signal['action'] in ['BUY', 'SELL'] and signal['confidence'] > 65:
                        logger.info(f"🎯 Entry signal detected: {signal['action']} with {signal['confidence']}% confidence")
                        self.open_position(signal)
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️  Bot stopped by user")
            if self.position:
                logger.warning("⚠️  Position still open! Close manually or restart bot")
        except Exception as e:
            logger.error(f"❌ Critical error: {e}")
            if self.position:
                logger.warning("⚠️  Position still open! Close manually")


def main():
    """Main entry point"""
    
    # Load configuration from environment variables
    config = {
        'api_key': os.getenv('BITGET_API_KEY'),
        'api_secret': os.getenv('BITGET_API_SECRET'),
        'passphrase': os.getenv('BITGET_PASSPHRASE'),
        'symbol': os.getenv('TRADING_SYMBOL', 'BTC/USDT:USDT'),
        'risk_percentage': float(os.getenv('RISK_PERCENTAGE', '10')),
        'check_interval': int(os.getenv('CHECK_INTERVAL', '30'))
    }
    
    # Validate configuration
    if not all([config['api_key'], config['api_secret'], config['passphrase']]):
        logger.error("❌ Missing API credentials in environment variables!")
        logger.error("   Set: BITGET_API_KEY, BITGET_API_SECRET, BITGET_PASSPHRASE")
        return
    
    # Create and run bot
    bot = BitgetFuturesBot(config)
    bot.run()


if __name__ == "__main__":
    main()
