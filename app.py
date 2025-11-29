import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BitgetFuturesBot:
    def __init__(self, api_key, api_secret, passphrase, symbol, risk_percentage, margin_mode='cross', leverage=10):
        try:
            self.exchange = ccxt.bitget({
                'apiKey': api_key,
                'secret': api_secret,
                'password': passphrase,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'defaultMarginMode': margin_mode
                }
            })
            
            self.symbol = symbol
            self.risk_percentage = risk_percentage
            self.margin_mode = margin_mode
            self.leverage = leverage
            self.position = None
            self.trades_history = []
            self.total_profit = 0
            self.is_running = False
            
            # Test connection
            self.exchange.fetch_balance()
            
            # Set leverage for the symbol
            try:
                self.exchange.set_leverage(self.leverage, self.symbol)
                logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
            except Exception as e:
                logger.warning(f"Could not set leverage: {e}")
            
            logger.info(f"Bot initialized: {symbol} | {risk_percentage}% risk | {leverage}x leverage | {margin_mode} margin")
            
        except Exception as e:
            logger.error(f"Bot initialization error: {e}")
            raise
    
    def fetch_ohlcv(self, timeframe='5m', limit=200):
        """Fetch OHLCV data with error handling"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators with error handling"""
        try:
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
            
            # RSI
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
            
            # Stochastic
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR
            atr = AverageTrueRange(df['high'], df['low'], df['close'])
            df['atr'] = atr.average_true_range()
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            
            # Volume
            df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_trend(self, df):
        """Analyze market trend"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            trend_score = 0
            
            # SMA alignment
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                trend_score += 3
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                trend_score -= 3
            
            # EMA crossover
            if latest['ema_12'] > latest['ema_26'] and prev['ema_12'] <= prev['ema_26']:
                trend_score += 2
            elif latest['ema_12'] < latest['ema_26'] and prev['ema_12'] >= prev['ema_26']:
                trend_score -= 2
            elif latest['ema_12'] > latest['ema_26']:
                trend_score += 1
            elif latest['ema_12'] < latest['ema_26']:
                trend_score -= 1
            
            # MACD
            if latest['macd'] > latest['macd_signal'] and latest['macd_diff'] > 0:
                trend_score += 1
            elif latest['macd'] < latest['macd_signal'] and latest['macd_diff'] < 0:
                trend_score -= 1
            
            return trend_score
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return 0
    
    def analyze_momentum(self, df):
        """Analyze momentum"""
        try:
            latest = df.iloc[-1]
            momentum_score = 0
            
            # RSI
            if latest['rsi'] < 30:
                momentum_score += 2
            elif latest['rsi'] > 70:
                momentum_score -= 2
            elif 40 < latest['rsi'] < 60:
                momentum_score += 0
            
            # Stochastic
            if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
                momentum_score += 1
            elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
                momentum_score -= 1
            
            return momentum_score
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return 0
    
    def analyze_volume(self, df):
        """Analyze volume"""
        try:
            latest = df.iloc[-1]
            volume_score = 0
            
            if latest['volume_ratio'] > 1.5:
                volume_score = 2
            elif latest['volume_ratio'] > 1.0:
                volume_score = 1
            else:
                volume_score = -1
            
            return volume_score
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return 0
    
    def generate_signal(self, df):
        """Generate trading signal - ALWAYS returns BUY or SELL, never HOLD"""
        try:
            trend_score = self.analyze_trend(df)
            momentum_score = self.analyze_momentum(df)
            volume_score = self.analyze_volume(df)
            
            total_score = trend_score + momentum_score + volume_score
            latest = df.iloc[-1]
            
            # Determine direction based on total score
            # Positive = BUY, Negative/Zero = SELL
            if total_score > 0:
                action = 'BUY'
                confidence = min(abs(total_score) * 10 + 50, 95)
            else:
                action = 'SELL'
                confidence = min(abs(total_score) * 10 + 50, 95)
            
            # If score is exactly 0, use RSI to decide
            if total_score == 0:
                if latest['rsi'] < 50:
                    action = 'BUY'
                    confidence = 50
                else:
                    action = 'SELL'
                    confidence = 50
            
            signal = {
                'action': action,
                'confidence': confidence,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'total_score': total_score,
                'price': latest['close'],
                'rsi': latest['rsi'],
                'atr_pct': latest['atr_pct']
            }
            
            return signal
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            # Default to SELL if error (safer than BUY)
            return {
                'action': 'SELL',
                'confidence': 50,
                'trend_score': 0,
                'momentum_score': 0,
                'volume_score': 0,
                'total_score': 0,
                'price': 0,
                'rsi': 50,
                'atr_pct': 0
            }
    
    def get_account_balance(self):
        """Get account balance with error handling"""
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0
    
    def calculate_position_size(self, price):
        """Calculate position size with leverage"""
        try:
            balance = self.get_account_balance()
            # Account for leverage in position size
            position_value = balance * (self.risk_percentage / 100) * self.leverage
            position_size = position_value / price
            return round(position_size, 4)
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def open_position(self, signal):
        """Open position with error handling - 1% TP, 10% SL"""
        try:
            size = self.calculate_position_size(signal['price'])
            
            if size <= 0:
                return False, "Invalid position size"
            
            if signal['action'] == 'BUY':
                order = self.exchange.create_market_buy_order(self.symbol, size)
                side = 'long'
            else:
                order = self.exchange.create_market_sell_order(self.symbol, size)
                side = 'short'
            
            # Calculate TP and SL prices (1% TP, 10% SL)
            entry_price = signal['price']
            if side == 'long':
                tp_price = entry_price * 1.01  # 1% TP
                sl_price = entry_price * 0.90  # 10% SL
            else:
                tp_price = entry_price * 0.99  # 1% TP
                sl_price = entry_price * 1.10  # 10% SL
            
            self.position = {
                'side': side,
                'entry_price': entry_price,
                'size': size,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'entry_time': datetime.now(),
                'signal': signal
            }
            
            logger.info(f"✅ Opened {side.upper()} position: {size} @ ${entry_price:.2f} | {self.leverage}x leverage")
            logger.info(f"   TP: ${tp_price:.2f} (+1%) | SL: ${sl_price:.2f} (-10%)")
            
            return True, f"✅ Opened {side.upper()} position\nSize: {size}\nEntry: ${entry_price:.2f}\nLeverage: {self.leverage}x\nTP: ${tp_price:.2f} (+1%)\nSL: ${sl_price:.2f} (-10%)"
            
        except Exception as e:
            error_msg = f"Error opening position: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, f"❌ {error_msg}"
    
    def close_position(self, current_price, reason=""):
        """Close position with error handling"""
        try:
            if not self.position:
                return False, "No open position"
            
            size = self.position['size']
            
            if self.position['side'] == 'long':
                order = self.exchange.create_market_sell_order(self.symbol, size)
                pnl = (current_price - self.position['entry_price']) * size
            else:
                order = self.exchange.create_market_buy_order(self.symbol, size)
                pnl = (self.position['entry_price'] - current_price) * size
            
            pnl_pct = (pnl / (self.position['entry_price'] * size / self.leverage)) * 100
            self.total_profit += pnl
            
            trade_record = {
                'side': self.position['side'],
                'entry_price': self.position['entry_price'],
                'exit_price': current_price,
                'size': size,
                'leverage': self.leverage,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'entry_time': self.position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'reason': reason
            }
            
            self.trades_history.append(trade_record)
            logger.info(f"🔒 Closed {self.position['side'].upper()} position")
            logger.info(f"   Entry: ${self.position['entry_price']:.2f} | Exit: ${current_price:.2f}")
            logger.info(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
            logger.info(f"   Total Profit: ${self.total_profit:.2f}")
            
            self.position = None
            return True, f"🔒 Position Closed\nP&L: ${pnl:.2f} ({pnl_pct:+.2f}%)\nReason: {reason}"
            
        except Exception as e:
            error_msg = f"Error closing position: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, f"❌ {error_msg}"
    
    def check_exit_conditions(self, df, current_price):
        """Check TP (1%) and SL (10%) conditions"""
        try:
            if not self.position:
                return False, ""
            
            entry_price = self.position['entry_price']
            side = self.position['side']
            tp_price = self.position['tp_price']
            sl_price = self.position['sl_price']
            
            # Calculate current PnL
            if side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                # Check TP (1%)
                if current_price >= tp_price:
                    return True, f"Take Profit 1% ({pnl_pct:.2f}%)"
                
                # Check SL (10%)
                if current_price <= sl_price:
                    return True, f"Stop Loss -10% ({pnl_pct:.2f}%)"
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Check TP (1%)
                if current_price <= tp_price:
                    return True, f"Take Profit 1% ({pnl_pct:.2f}%)"
                
                # Check SL (10%)
                if current_price >= sl_price:
                    return True, f"Stop Loss -10% ({pnl_pct:.2f}%)"
            
            return False, f"Holding ({pnl_pct:+.2f}%)"
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False, ""


# Global bot instance
bot_instance = None

# Initialize Dash app with dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

# Custom dark theme colors
DARK_BG = '#1e1e1e'
CARD_BG = '#2d2d2d'
TEXT_COLOR = '#ffffff'
SUCCESS_COLOR = '#00ff00'
DANGER_COLOR = '#ff0000'
PRIMARY_COLOR = '#00bcd4'

# App layout
app.layout = dbc.Container([
    dcc.Interval(id='interval-component', interval=30000, n_intervals=0, disabled=True),
    dcc.Store(id='bot-state', data={'running': False, 'initialized': False}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🤖 Bitget Futures Trading Bot", 
                   style={'color': PRIMARY_COLOR, 'textAlign': 'center', 'marginTop': '20px'}),
            html.H5("10x Leverage • 1% TP • 10% SL • Always Trading Mode", 
                   style={'color': TEXT_COLOR, 'textAlign': 'center', 'marginBottom': '20px'})
        ])
    ]),
    
    html.Hr(style={'borderColor': PRIMARY_COLOR}),
    
    # Configuration Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("⚙️ Bot Configuration", style={'color': PRIMARY_COLOR})),
                dbc.CardBody([
                    # API Credentials
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("API Key", style={'color': TEXT_COLOR}),
                            dbc.Input(id='api-key', type='password', placeholder='Enter API Key')
                        ], width=4),
                        dbc.Col([
                            dbc.Label("API Secret", style={'color': TEXT_COLOR}),
                            dbc.Input(id='api-secret', type='password', placeholder='Enter API Secret')
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Passphrase", style={'color': TEXT_COLOR}),
                            dbc.Input(id='passphrase', type='password', placeholder='Enter Passphrase')
                        ], width=4),
                    ], className='mb-3'),
                    
                    # Trading Settings
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Trading Symbol", style={'color': TEXT_COLOR}),
                            dcc.Dropdown(
                                id='symbol-dropdown',
                                options=[
                                    {'label': '₿ Bitcoin (BTC)', 'value': 'BTC/USDT:USDT'},
                                    {'label': '⟠ Ethereum (ETH)', 'value': 'ETH/USDT:USDT'},
                                    {'label': '◎ Solana (SOL)', 'value': 'SOL/USDT:USDT'},
                                    {'label': '◆ Binance Coin (BNB)', 'value': 'BNB/USDT:USDT'},
                                    {'label': '✕ Ripple (XRP)', 'value': 'XRP/USDT:USDT'},
                                    {'label': '💎 Toncoin (TON)', 'value': 'TON/USDT:USDT'}
                                ],
                                value='BTC/USDT:USDT',
                                style={'color': '#000'}
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Account % per Trade", style={'color': TEXT_COLOR}),
                            dbc.Input(id='risk-percentage', type='number', value=5, min=1, max=20, step=1)
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Margin Mode", style={'color': TEXT_COLOR}),
                            dcc.Dropdown(
                                id='margin-mode',
                                options=[
                                    {'label': 'Cross Margin', 'value': 'cross'},
                                    {'label': 'Isolated Margin', 'value': 'isolated'}
                                ],
                                value='cross',
                                style={'color': '#000'}
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Strategy", style={'color': TEXT_COLOR}),
                            html.Div([
                                html.Small("10x Leverage", 
                                         style={'color': PRIMARY_COLOR, 'fontWeight': 'bold', 'display': 'block'}),
                                html.Small("TP: 1% | SL: 10%", 
                                         style={'color': SUCCESS_COLOR, 'fontWeight': 'bold'})
                            ])
                        ], width=3),
                    ], className='mb-3'),
                    
                    # Control Buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("🚀 Start Bot", id='start-button', color='success', 
                                     className='w-100', size='lg')
                        ], width=6),
                        dbc.Col([
                            dbc.Button("🛑 Stop Bot", id='stop-button', color='danger', 
                                     className='w-100', size='lg')
                        ], width=6),
                    ]),
                    
                    html.Div(id='status-message', className='mt-3')
                ])
            ], style={'backgroundColor': CARD_BG}, className='mb-4')
        ], width=12)
    ]),
    
    # Metrics Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("💵 Current Price", style={'color': TEXT_COLOR}),
                    html.H3(id='current-price', children='$0.00', 
                           style={'color': PRIMARY_COLOR})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("💰 Total Profit", style={'color': TEXT_COLOR}),
                    html.H3(id='total-profit', children='$0.00', 
                           style={'color': SUCCESS_COLOR})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("💳 Balance", style={'color': TEXT_COLOR}),
                    html.H3(id='balance', children='$0.00', 
                           style={'color': TEXT_COLOR})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📊 Signal", style={'color': TEXT_COLOR}),
                    html.H3(id='signal', children='WAITING', 
                           style={'color': TEXT_COLOR})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📈 Total Trades", style={'color': TEXT_COLOR}),
                    html.H3(id='total-trades', children='0', 
                           style={'color': TEXT_COLOR})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🎯 Win Rate", style={'color': TEXT_COLOR}),
                    html.H3(id='win-rate', children='0%', 
                           style={'color': TEXT_COLOR})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=2),
    ], className='mb-4'),
    
    # Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📈 Technical Analysis", style={'color': PRIMARY_COLOR})),
                dbc.CardBody([
                    dcc.Graph(id='price-chart', style={'height': '600px'})
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=12)
    ], className='mb-4'),
    
    # Position Info
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Active Position (10x Leverage)", style={'color': PRIMARY_COLOR})),
                dbc.CardBody(id='position-info')
            ], style={'backgroundColor': CARD_BG})
        ], width=12)
    ], className='mb-4'),
    
    # Trade History
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📋 Trade History", style={'color': PRIMARY_COLOR})),
                dbc.CardBody([
                    html.Div(id='trade-history')
                ])
            ], style={'backgroundColor': CARD_BG})
        ], width=12)
    ])
    
], fluid=True, style={'backgroundColor': DARK_BG, 'minHeight': '100vh', 'padding': '20px'})


# Callbacks
@app.callback(
    [Output('bot-state', 'data'),
     Output('status-message', 'children'),
     Output('interval-component', 'disabled')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('api-key', 'value'),
     State('api-secret', 'value'),
     State('passphrase', 'value'),
     State('symbol-dropdown', 'value'),
     State('risk-percentage', 'value'),
     State('margin-mode', 'value'),
     State('bot-state', 'data')]
)
def control_bot(start_clicks, stop_clicks, api_key, api_secret, passphrase, 
                symbol, risk_pct, margin_mode, current_state):
    global bot_instance
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_state, "", True
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and start_clicks:
        # Validate inputs
        if not all([api_key, api_secret, passphrase]):
            return current_state, dbc.Alert("❌ Please enter all API credentials", color="danger"), True
        
        try:
            # Initialize bot with 10x leverage
            bot_instance = BitgetFuturesBot(api_key, api_secret, passphrase, symbol, risk_pct, margin_mode, leverage=10)
            bot_instance.is_running = True
            
            return {
                'running': True,
                'initialized': True
            }, dbc.Alert(f"✅ Bot started! Trading {symbol} with {risk_pct}% risk, 10x leverage, {margin_mode} margin | TP: 1% | SL: 10%", 
                        color="success"), False
            
        except Exception as e:
            error_msg = f"Failed to initialize bot: {str(e)}"
            logger.error(error_msg)
            return current_state, dbc.Alert(f"❌ {error_msg}", color="danger"), True
    
    elif button_id == 'stop-button' and stop_clicks:
        if bot_instance:
            bot_instance.is_running = False
        
        return {
            'running': False,
            'initialized': False
        }, dbc.Alert("🛑 Bot stopped", color="warning"), True
    
    return current_state, "", True


@app.callback(
    [Output('current-price', 'children'),
     Output('total-profit', 'children'),
     Output('balance', 'children'),
     Output('signal', 'children'),
     Output('total-trades', 'children'),
     Output('win-rate', 'children'),
     Output('price-chart', 'figure'),
     Output('position-info', 'children'),
     Output('trade-history', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('bot-state', 'data')]
)
def update_dashboard(n_intervals, bot_state):
    global bot_instance
    
    if not bot_state.get('running') or not bot_instance:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        return '$0.00', '$0.00', '$0.00', 'WAITING', '0', '0%', empty_fig, "No active position", "No trades yet"
    
    try:
        # Fetch data
        df = bot_instance.fetch_ohlcv()
        if df is None:
            raise Exception("Failed to fetch market data")
        
        df = bot_instance.calculate_indicators(df)
        signal = bot_instance.generate_signal(df)
        current_price = signal['price']
        balance = bot_instance.get_account_balance()
        
        # Trading logic with detailed logging
        if bot_instance.position:
            logger.info(f"Active position: {bot_instance.position['side']} at ${bot_instance.position['entry_price']:.2f}")
            should_close, reason = bot_instance.check_exit_conditions(df, current_price)
            logger.info(f"Exit check: {should_close} - {reason}")
            if should_close:
                success, message = bot_instance.close_position(current_price, reason)
                logger.info(f"Close result: {message}")
        else:
            logger.info(f"No position. Checking for entry signal...")
            logger.info(f"Signal: {signal['action']} | Confidence: {signal['confidence']}% | "
                       f"Trend: {signal['trend_score']} | Momentum: {signal['momentum_score']} | "
                       f"Volume: {signal['volume_score']} | Total: {signal['total_score']}")
            
            # ALWAYS trade - no minimum confidence requirement
            logger.info(f"🚀 Opening {signal['action']} position (always trading mode with 10x leverage)...")
            success, message = bot_instance.open_position(signal)
            logger.info(f"Open result: Success={success} | {message}")
            
            if not success:
                logger.error(f"Failed to open position: {message}")
        
        # Create chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price Action', 'RSI', 'MACD')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
        
        # SMAs
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_20'], name='SMA 20', 
                                line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sma_50'], name='SMA 50', 
                                line=dict(color='orange', width=1)), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_upper'], name='BB Upper', 
                                line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['bb_lower'], name='BB Lower', 
                                line=dict(color='gray', width=1, dash='dash'), 
                                fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        # Mark entry if position exists
        if bot_instance.position:
            entry_time = bot_instance.position['entry_time']
            entry_price = bot_instance.position['entry_price']
            tp_price = bot_instance.position['tp_price']
            sl_price = bot_instance.position['sl_price']
            color = 'green' if bot_instance.position['side'] == 'long' else 'red'
            
            # Entry point
            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol='star'),
                    name=f"Entry ({bot_instance.position['side'].upper()})",
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # TP and SL lines
            fig.add_hline(y=tp_price, line_dash="dash", line_color="green", 
                         annotation_text=f"TP 1%: ${tp_price:.2f}", 
                         annotation_position="right", row=1, col=1)
            fig.add_hline(y=sl_price, line_dash="dash", line_color="red", 
                         annotation_text=f"SL -10%: ${sl_price:.2f}", 
                         annotation_position="right", row=1, col=1)
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI', 
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['macd'], name='MACD', 
                      line=dict(color='blue', width=1)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['macd_signal'], name='Signal', 
                      line=dict(color='red', width=1)),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['macd_diff'], 
                  name='Histogram', marker_color=colors, opacity=0.5),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified',
            height=600
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        # Position info
        position_info = html.Div([
            html.P("No active position - Next trade opening shortly...", 
                  style={'color': TEXT_COLOR, 'textAlign': 'center'})
        ])
        
        if bot_instance.position:
            pos = bot_instance.position
            side = pos['side']
            entry_price = pos['entry_price']
            
            if side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 * bot_instance.leverage
                pnl_usd = (current_price - entry_price) * pos['size']
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100 * bot_instance.leverage
                pnl_usd = (entry_price - current_price) * pos['size']
            
            pnl_color = SUCCESS_COLOR if pnl_usd > 0 else DANGER_COLOR
            side_color = SUCCESS_COLOR if side == 'long' else DANGER_COLOR
            
            position_info = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H6("Side", style={'color': TEXT_COLOR}),
                        html.H4(side.upper(), style={'color': side_color, 'fontWeight': 'bold'})
                    ], width=2),
                    dbc.Col([
                        html.H6("Entry Price", style={'color': TEXT_COLOR}),
                        html.H4(f"${entry_price:.2f}", style={'color': TEXT_COLOR})
                    ], width=2),
                    dbc.Col([
                        html.H6("Current Price", style={'color': TEXT_COLOR}),
                        html.H4(f"${current_price:.2f}", style={'color': TEXT_COLOR})
                    ], width=2),
                    dbc.Col([
                        html.H6("Size", style={'color': TEXT_COLOR}),
                        html.H4(f"{pos['size']}", style={'color': TEXT_COLOR})
                    ], width=1),
                    dbc.Col([
                        html.H6("Leverage", style={'color': TEXT_COLOR}),
                        html.H4(f"{bot_instance.leverage}x", style={'color': PRIMARY_COLOR, 'fontWeight': 'bold'})
                    ], width=1),
                    dbc.Col([
                        html.H6("Take Profit", style={'color': TEXT_COLOR}),
                        html.H4(f"${pos['tp_price']:.2f}", style={'color': SUCCESS_COLOR})
                    ], width=2),
                    dbc.Col([
                        html.H6("Stop Loss", style={'color': TEXT_COLOR}),
                        html.H4(f"${pos['sl_price']:.2f}", style={'color': DANGER_COLOR})
                    ], width=2),
                ], className='mb-3'),
                dbc.Row([
                    dbc.Col([
                        html.H6("Unrealized P&L", style={'color': TEXT_COLOR, 'textAlign': 'center'}),
                        html.H2(f"${pnl_usd:.2f}", 
                               style={'color': pnl_color, 'fontWeight': 'bold', 'textAlign': 'center'}),
                        html.H4(f"({pnl_pct:+.2f}%)", 
                               style={'color': pnl_color, 'textAlign': 'center'})
                    ], width=12)
                ])
            ])
        
        # Trade history
        trade_history = html.P("No trades yet - First trade will open shortly...", 
                              style={'color': TEXT_COLOR, 'textAlign': 'center'})
        
        if bot_instance.trades_history:
            trades_df = pd.DataFrame(bot_instance.trades_history)
            
            # Statistics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Recent trades table
            display_df = trades_df[['entry_time', 'side', 'entry_price', 'exit_price', 
                                   'leverage', 'pnl', 'pnl_pct', 'reason']].tail(15)
            
            # Format numbers
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
            
            trade_history = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H6("Total Trades", style={'color': TEXT_COLOR}),
                        html.H4(str(total_trades), style={'color': PRIMARY_COLOR, 'fontWeight': 'bold'})
                    ], width=2),
                    dbc.Col([
                        html.H6("Winners", style={'color': TEXT_COLOR}),
                        html.H4(str(winning_trades), style={'color': SUCCESS_COLOR, 'fontWeight': 'bold'})
                    ], width=2),
                    dbc.Col([
                        html.H6("Losers", style={'color': TEXT_COLOR}),
                        html.H4(str(losing_trades), style={'color': DANGER_COLOR, 'fontWeight': 'bold'})
                    ], width=2),
                    dbc.Col([
                        html.H6("Win Rate", style={'color': TEXT_COLOR}),
                        html.H4(f"{win_rate:.1f}%", style={'color': PRIMARY_COLOR, 'fontWeight': 'bold'})
                    ], width=2),
                    dbc.Col([
                        html.H6("Avg Win", style={'color': TEXT_COLOR}),
                        html.H4(f"${avg_win:.2f}", style={'color': SUCCESS_COLOR})
                    ], width=2),
                    dbc.Col([
                        html.H6("Avg Loss", style={'color': TEXT_COLOR}),
                        html.H4(f"${avg_loss:.2f}", style={'color': DANGER_COLOR})
                    ], width=2),
                ], className='mb-3'),
                
                html.H6("Recent Trades", style={'color': TEXT_COLOR, 'marginTop': '20px'}),
                dash_table.DataTable(
                    data=display_df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in display_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': CARD_BG,
                        'color': TEXT_COLOR,
                        'border': '1px solid #444',
                        'textAlign': 'left',
                        'padding': '10px',
                        'fontSize': '12px'
                    },
                    style_header={
                        'backgroundColor': DARK_BG,
                        'color': PRIMARY_COLOR,
                        'fontWeight': 'bold',
                        'border': '1px solid #444'
                    },
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{pnl} contains "$-" || {pnl} contains "$0.00"',
                            },
                            'backgroundColor': 'rgba(255, 0, 0, 0.2)',
                        },
                        {
                            'if': {
                                'filter_query': '{pnl} contains "$" && {pnl} does not contain "$-" && {pnl} does not contain "$0.00"',
                            },
                            'backgroundColor': 'rgba(0, 255, 0, 0.2)',
                        }
                    ],
                    page_size=15
                )
            ])
        
        # Signal display
        signal_color = SUCCESS_COLOR if signal['action'] == 'BUY' else DANGER_COLOR
        signal_text = f"{signal['action']} ({signal['confidence']}%)"
        
        # Calculate win rate
        win_rate_text = '0%'
        if bot_instance.trades_history:
            trades = pd.DataFrame(bot_instance.trades_history)
            winners = len(trades[trades['pnl'] > 0])
            win_rate_text = f"{(winners/len(trades)*100):.1f}%"
        
        # Format total profit color
        profit_color = SUCCESS_COLOR if bot_instance.total_profit > 0 else DANGER_COLOR if bot_instance.total_profit < 0 else TEXT_COLOR
        
        return (
            f"${current_price:.2f}",
            html.Span(f"${bot_instance.total_profit:.2f}", style={'color': profit_color}),
            f"${balance:.2f}",
            html.Span(signal_text, style={'color': signal_color, 'fontWeight': 'bold'}),
            str(len(bot_instance.trades_history)),
            win_rate_text,
            fig,
            position_info,
            trade_history
        )
        
    except Exception as e:
        logger.error(f"Dashboard update error: {e}")
        logger.error(traceback.format_exc())
        
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor=CARD_BG,
            plot_bgcolor=CARD_BG,
            title="Error loading chart"
        )
        
        error_msg = html.Div([
            dbc.Alert(f"⚠️ Error: {str(e)}", color="danger")
        ])
        
        return (
            '$0.00',
            '$0.00',
            '$0.00',
            'ERROR',
            '0',
            '0%',
            empty_fig,
            error_msg,
            error_msg
        )


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
