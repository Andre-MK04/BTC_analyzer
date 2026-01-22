from __future__ import annotations

import math
import os
import json
import smtplib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_btc_data(start: str = "2014-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetch BTC-USD daily data from Yahoo Finance."""
    df = yf.download("BTC-USD", start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return a MultiIndex even for a single ticker
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators efficiently."""
    df = df.copy()

    def _to_series(col: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(col, pd.DataFrame):
            return col.iloc[:, 0]
        return col

    close = _to_series(df["Close"])
    high = _to_series(df["High"])
    low = _to_series(df["Low"])
    volume = _to_series(df["Volume"])
    
    # Trend indicators
    df["EMA12"] = EMAIndicator(close, window=12).ema_indicator()
    df["EMA26"] = EMAIndicator(close, window=26).ema_indicator()
    df["EMA50"] = EMAIndicator(close, window=50).ema_indicator()
    df["EMA200"] = EMAIndicator(close, window=200).ema_indicator()
    
    # Momentum
    df["RSI"] = RSIIndicator(close, window=14).rsi()
    df["RSI_MA"] = df["RSI"].rolling(window=5).mean()
    
    # Volatility
    df["ATR"] = AverageTrueRange(high, low, close, window=14).average_true_range()
    df["ATR_pct"] = (df["ATR"] / close) * 100
    
    # Historical volatility (for Kelly)
    df["Returns"] = close.pct_change()
    df["Volatility_20"] = df["Returns"].rolling(window=20).std() * np.sqrt(365)
    
    # Bollinger Bands
    bb = BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    
    # ADX for trend strength
    adx = ADXIndicator(high, low, close, window=14)
    df["ADX"] = adx.adx()
    df["DI_plus"] = adx.adx_pos()
    df["DI_minus"] = adx.adx_neg()
    
    # Price metrics
    df["High_20"] = high.rolling(window=20).max()
    df["Low_20"] = low.rolling(window=20).min()
    df["High_50"] = high.rolling(window=50).max()
    df["Low_50"] = low.rolling(window=50).min()
    
    # Volume
    df["Volume_MA"] = volume.rolling(window=20).mean()
    df["Volume_ratio"] = volume / df["Volume_MA"]
    
    return df


# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Detect market regime to enable/disable strategies.
    - Trending: Strong directional movement (ADX high, price away from MA)
    - Ranging: Choppy, mean-reverting (ADX low, BB tight)
    - Volatile: High ATR, wide BBs (reduce position sizes)
    """
    
    @staticmethod
    def detect(row: pd.Series) -> str:
        """Detect current market regime."""
        adx = row.get("ADX", np.nan)
        bb_width = row.get("BB_width", np.nan)
        atr_pct = row.get("ATR_pct", np.nan)
        
        if pd.isna([adx, bb_width, atr_pct]).any():
            return "unknown"
        
        # Volatile regime (reduce all position sizes)
        if atr_pct > 8:
            return "volatile"
        
        # Trending regime (favor trend-following)
        if adx > 30 and bb_width > 0.1:
            return "trending"
        
        # Ranging regime (favor mean-reversion)
        if adx < 20 and bb_width < 0.08:
            return "ranging"
        
        # Mixed/transition
        if 20 <= adx <= 30:
            return "mixed"
        
        return "unknown"
    
    @staticmethod
    def get_strategy_weights(regime: str) -> Dict[str, float]:
        """Get strategy weight multipliers based on regime."""
        weights = {
            "trending": {"TrendFollowing": 1.5, "MeanReversion": 0.3, "Breakout": 1.0},
            "ranging": {"TrendFollowing": 0.4, "MeanReversion": 1.5, "Breakout": 0.7},
            "mixed": {"TrendFollowing": 1.0, "MeanReversion": 1.0, "Breakout": 1.0},
            "volatile": {"TrendFollowing": 0.7, "MeanReversion": 0.5, "Breakout": 0.8},
            "unknown": {"TrendFollowing": 0.8, "MeanReversion": 0.8, "Breakout": 0.8},
        }
        return weights.get(regime, weights["mixed"])


# =============================================================================
# KELLY CRITERION CALCULATOR
# =============================================================================

class KellyCalculator:
    """
    Calculate optimal position size using Kelly Criterion.
    Kelly% = (Win% * AvgWin - Loss% * AvgLoss) / AvgWin
    
    Uses fractional Kelly (typically 0.25-0.5) to reduce volatility.
    """
    
    def __init__(self, fraction: float = 0.25, lookback_trades: int = 30):
        self.fraction = fraction
        self.lookback_trades = lookback_trades
        self.trade_history: List[Dict] = []
    
    def add_trade(self, pnl: float, risk: float):
        """Record a completed trade."""
        self.trade_history.append({"pnl": pnl, "risk": risk})
    
    def calculate_kelly(self, strategy: str = None) -> float:
        """
        Calculate Kelly percentage for strategy.
        Returns value between 0 and 1 (fraction of capital).
        """
        if len(self.trade_history) < 10:
            # Not enough history, use conservative default
            return 0.2 * self.fraction
        
        # Use recent trades
        recent = self.trade_history[-self.lookback_trades:]
        
        if strategy:
            # Filter by strategy if provided
            recent = [t for t in recent if t.get("strategy") == strategy]
            if len(recent) < 5:
                return 0.2 * self.fraction
        
        pnls = [t["pnl"] for t in recent]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        if not wins or not losses:
            return 0.2 * self.fraction
        
        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.2 * self.fraction
        
        # Kelly formula
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply fraction and clip to reasonable bounds
        kelly = kelly * self.fraction
        kelly = np.clip(kelly, 0.05, 0.5)
        
        return float(kelly)


# =============================================================================
# STRATEGY COMPONENTS
# =============================================================================

@dataclass
class Signal:
    """Trading signal with risk parameters."""
    date: pd.Timestamp
    direction: int  # 1=long, -1=short, 0=flat
    strategy: str
    confidence: float  # 0 to 1
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # fraction of capital
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TrendFollowingStrategy:
    """
    Classic trend following with EMA crossovers.
    Uses trailing stops to capture extended moves.
    """
    
    def __init__(
        self,
        atr_stop_mult: float = 2.0,
        trailing_stop_activation: float = 1.5,  # Activate after 1.5x ATR profit
        trailing_stop_distance: float = 1.0,  # Trail by 1x ATR
    ):
        self.name = "TrendFollowing"
        self.atr_stop_mult = atr_stop_mult
        self.trailing_stop_activation = trailing_stop_activation
        self.trailing_stop_distance = trailing_stop_distance
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: str,
        kelly_size: float
    ) -> Optional[Signal]:
        """Generate signal for given bar index."""
        if idx < 200:
            return None
        
        # Check if regime allows this strategy
        regime_weights = RegimeDetector.get_strategy_weights(regime)
        regime_mult = regime_weights.get(self.name, 1.0)
        
        if regime_mult < 0.5:  # Don't trade if heavily penalized
            return None
        
        row = df.iloc[idx]
        date = df.index[idx]
        
        close = row["Close"]
        ema12 = row["EMA12"]
        ema26 = row["EMA26"]
        ema50 = row["EMA50"]
        ema200 = row["EMA200"]
        rsi = row["RSI"]
        adx = row["ADX"]
        atr = row["ATR"]
        
        if pd.isna([ema12, ema26, ema50, ema200, rsi, adx, atr]).any():
            return None
        
        # Strong trend required
        trend_up = ema12 > ema26 > ema50 > ema200 and adx > 25
        trend_down = ema12 < ema26 < ema50 < ema200 and adx > 25
        
        if not (trend_up or trend_down):
            return None
        
        # Entry on pullback
        if trend_up:
            if 35 < rsi < 55 and close > ema50:
                direction = 1
                stop = close - self.atr_stop_mult * atr
                target = close + 4 * self.atr_stop_mult * atr  # Wider target for trending
                confidence = min(1.0, (adx - 25) / 40 + 0.4)
                use_trailing = True
            else:
                return None
        
        elif trend_down:
            if 45 < rsi < 65 and close < ema50:
                direction = -1
                stop = close + self.atr_stop_mult * atr
                target = close - 4 * self.atr_stop_mult * atr
                confidence = min(1.0, (adx - 25) / 40 + 0.4)
                use_trailing = True
            else:
                return None
        
        # Position sizing
        atr_pct = row["ATR_pct"]
        base_size = kelly_size
        
        # Adjust by volatility
        if atr_pct > 6:
            base_size *= 0.6
        elif atr_pct > 4:
            base_size *= 0.8
        
        # Adjust by regime
        base_size *= regime_mult
        
        # Adjust by confidence
        base_size *= confidence
        
        base_size = np.clip(base_size, 0.05, 0.5)
        
        return Signal(
            date=date,
            direction=direction,
            strategy=self.name,
            confidence=confidence,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            position_size=base_size,
            use_trailing_stop=use_trailing,
            trailing_stop_pct=self.trailing_stop_distance * atr / close * 100,
            metadata={
                "adx": adx,
                "rsi": rsi,
                "atr_pct": atr_pct,
                "regime": regime,
            }
        )


class MeanReversionStrategy:
    """
    Mean reversion for range-bound markets.
    Disabled in strong trending regimes.
    """
    
    def __init__(self):
        self.name = "MeanReversion"
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: str,
        kelly_size: float
    ) -> Optional[Signal]:
        """Generate signal for given bar index."""
        if idx < 200:
            return None
        
        # Check regime - don't trade mean reversion in trending markets
        regime_weights = RegimeDetector.get_strategy_weights(regime)
        regime_mult = regime_weights.get(self.name, 1.0)
        
        if regime_mult < 0.5:
            return None
        
        row = df.iloc[idx]
        date = df.index[idx]
        
        close = row["Close"]
        bb_upper = row["BB_upper"]
        bb_lower = row["BB_lower"]
        bb_middle = row["BB_middle"]
        bb_width = row["BB_width"]
        adx = row["ADX"]
        rsi = row["RSI"]
        atr = row["ATR"]
        
        if pd.isna([bb_upper, bb_lower, bb_middle, adx, rsi, atr]).any():
            return None
        
        # Only in non-trending markets
        if adx > 25:
            return None
        
        # Check for extremes
        at_lower = close <= bb_lower * 1.01
        at_upper = close >= bb_upper * 0.99
        
        if not (at_lower or at_upper):
            return None
        
        if at_lower and rsi < 30:
            direction = 1
            stop = close - 1.5 * atr
            target = bb_middle
            confidence = min(1.0, (30 - rsi) / 20 + 0.5)
        
        elif at_upper and rsi > 70:
            direction = -1
            stop = close + 1.5 * atr
            target = bb_middle
            confidence = min(1.0, (rsi - 70) / 20 + 0.5)
        
        else:
            return None
        
        # Position sizing
        base_size = kelly_size * 0.7  # More conservative for mean reversion
        base_size *= regime_mult
        base_size *= confidence
        base_size = np.clip(base_size, 0.05, 0.3)
        
        return Signal(
            date=date,
            direction=direction,
            strategy=self.name,
            confidence=confidence,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            position_size=base_size,
            use_trailing_stop=False,
            metadata={"adx": adx, "rsi": rsi, "bb_width": bb_width, "regime": regime}
        )


class BreakoutStrategy:
    """
    Breakout strategy for consolidation breaks.
    """
    
    def __init__(self, lookback: int = 20):
        self.name = "Breakout"
        self.lookback = lookback
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: str,
        kelly_size: float
    ) -> Optional[Signal]:
        """Generate signal for given bar index."""
        if idx < self.lookback + 50:
            return None
        
        regime_weights = RegimeDetector.get_strategy_weights(regime)
        regime_mult = regime_weights.get(self.name, 1.0)
        
        if regime_mult < 0.5:
            return None
        
        row = df.iloc[idx]
        date = df.index[idx]
        
        close = row["Close"]
        high_20 = row["High_20"]
        low_20 = row["Low_20"]
        volume_ratio = row["Volume_ratio"]
        atr = row["ATR"]
        bb_width = row["BB_width"]
        adx = row["ADX"]
        
        if pd.isna([high_20, low_20, volume_ratio, atr, bb_width]).any():
            return None
        
        # Need consolidation before breakout
        if bb_width > 0.15:
            return None
        
        # Volume confirmation
        if volume_ratio < 1.3:
            return None
        
        # Check for breakout
        prev_close = df.iloc[idx - 1]["Close"]
        
        if close > high_20 and prev_close <= high_20:
            direction = 1
            stop = low_20
            target = close + 2.5 * (close - low_20)
            confidence = min(1.0, volume_ratio / 2.5)
        
        elif close < low_20 and prev_close >= low_20:
            direction = -1
            stop = high_20
            target = close - 2.5 * (high_20 - close)
            confidence = min(1.0, volume_ratio / 2.5)
        
        else:
            return None
        
        # Position sizing
        base_size = kelly_size
        base_size *= regime_mult
        base_size *= confidence
        base_size = np.clip(base_size, 0.05, 0.4)
        
        return Signal(
            date=date,
            direction=direction,
            strategy=self.name,
            confidence=confidence,
            entry_price=close,
            stop_loss=stop,
            take_profit=target,
            position_size=base_size,
            use_trailing_stop=False,
            metadata={
                "volume_ratio": volume_ratio,
                "bb_width": bb_width,
                "adx": adx,
                "regime": regime
            }
        )


# =============================================================================
# POSITION MANAGEMENT WITH TRAILING STOPS
# =============================================================================

@dataclass
class Position:
    """Position with trailing stop support."""
    entry_date: pd.Timestamp
    entry_price: float
    direction: int
    size: float
    stop_loss: float
    take_profit: float
    strategy: str
    entry_capital: float
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    trailing_stop_price: float = 0.0
    
    def __post_init__(self):
        if self.use_trailing_stop:
            self.highest_price = self.entry_price
            self.lowest_price = self.entry_price
            self.trailing_stop_price = self.stop_loss
    
    def update_trailing_stop(self, current_price: float) -> None:
        """Update trailing stop if conditions are met."""
        if not self.use_trailing_stop:
            return
        
        if self.direction == 1:  # Long position
            # Update highest price seen
            if current_price > self.highest_price:
                self.highest_price = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 - self.trailing_stop_pct / 100)
                
                # Only raise the stop, never lower it
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
        
        else:  # Short position
            # Update lowest price seen
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                
                # Calculate new trailing stop
                new_stop = current_price * (1 + self.trailing_stop_pct / 100)
                
                # Only lower the stop, never raise it
                if new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
    
    def compute_pnl(self, current_price: float) -> float:
        """Compute current P&L in capital terms."""
        if self.direction == 1:
            return_pct = (current_price / self.entry_price) - 1
        else:
            return_pct = (self.entry_price / current_price) - 1
        
        return return_pct * self.size * self.entry_capital
    
    def check_exit(self, current_price: float) -> Tuple[bool, str]:
        """Check if position should be exited."""
        # Update trailing stop first
        self.update_trailing_stop(current_price)
        
        if self.direction == 1:
            # Check trailing stop if active
            if self.use_trailing_stop and current_price <= self.trailing_stop_price:
                return True, "trailing_stop"
            
            # Check fixed stop
            if current_price <= self.stop_loss:
                return True, "stop_loss"
            
            # Check take profit
            if current_price >= self.take_profit:
                return True, "take_profit"
        
        else:
            # Check trailing stop if active
            if self.use_trailing_stop and current_price >= self.trailing_stop_price:
                return True, "trailing_stop"
            
            # Check fixed stop
            if current_price >= self.stop_loss:
                return True, "stop_loss"
            
            # Check take profit
            if current_price <= self.take_profit:
                return True, "take_profit"
        
        return False, ""


# =============================================================================
# PORTFOLIO MANAGEMENT
# =============================================================================

class Portfolio:
    """Portfolio manager with Kelly sizing and risk controls."""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        max_positions: int = 3,
        max_risk_per_trade: float = 0.02,
        max_total_exposure: float = 1.0,
        fee_bps: float = 10.0,
        kelly_calculator: Optional[KellyCalculator] = None,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_positions = max_positions
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_exposure = max_total_exposure
        self.fee = fee_bps / 10000.0
        self.kelly = kelly_calculator or KellyCalculator()
        
        self.positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []
    
    def get_total_exposure(self) -> float:
        """Calculate current total position exposure."""
        return sum(p.size for p in self.positions)
    
    def can_add_position(self, signal: Signal) -> bool:
        """Check if new position can be added."""
        if len(self.positions) >= self.max_positions:
            return False
        
        if self.get_total_exposure() + signal.position_size > self.max_total_exposure:
            return False
        
        return True
    
    def adjust_position_size(self, signal: Signal) -> float:
        """Adjust position size based on risk management and Kelly."""
        # Calculate risk in price terms
        risk_per_share = abs(signal.entry_price - signal.stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Position size based on max risk per trade
        max_loss = self.capital * self.max_risk_per_trade
        shares_by_risk = max_loss / risk_per_share
        position_value = shares_by_risk * signal.entry_price
        size_by_risk = position_value / self.capital
        
        # Take minimum of signal size and risk-based size
        final_size = min(signal.position_size, size_by_risk)
        
        # Ensure we don't exceed max exposure
        available_exposure = self.max_total_exposure - self.get_total_exposure()
        final_size = min(final_size, available_exposure)
        
        return max(0, final_size)
    
    def open_position(self, signal: Signal) -> bool:
        """Open new position from signal."""
        if not self.can_add_position(signal):
            return False
        
        adjusted_size = self.adjust_position_size(signal)
        
        if adjusted_size < 0.05:
            return False
        
        # Deduct fees
        fee_cost = adjusted_size * self.capital * self.fee
        self.capital -= fee_cost
        
        pos = Position(
            entry_date=signal.date,
            entry_price=signal.entry_price,
            direction=signal.direction,
            size=adjusted_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            entry_capital=self.capital,
            use_trailing_stop=signal.use_trailing_stop,
            trailing_stop_pct=signal.trailing_stop_pct,
        )
        
        self.positions.append(pos)
        
        self.trade_log.append({
            "date": signal.date,
            "action": "OPEN",
            "strategy": signal.strategy,
            "direction": "LONG" if signal.direction == 1 else "SHORT",
            "price": signal.entry_price,
            "size": adjusted_size,
            "stop": signal.stop_loss,
            "target": signal.take_profit,
            "confidence": signal.confidence,
            "fee": fee_cost,
            "trailing_stop": signal.use_trailing_stop,
            **signal.metadata,
        })
        
        return True
    
    def update_positions(self, date: pd.Timestamp, current_price: float) -> None:
        """Update all positions and check for exits."""
        positions_to_remove = []
        
        for i, pos in enumerate(self.positions):
            should_exit, reason = pos.check_exit(current_price)
            
            if should_exit:
                self.close_position(i, date, current_price, reason)
                positions_to_remove.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        # Update equity
        total_equity = self.capital
        for pos in self.positions:
            total_equity += pos.compute_pnl(current_price)
        
        self.equity_curve.append(total_equity)
    
    def close_position(
        self,
        pos_idx: int,
        date: pd.Timestamp,
        price: float,
        reason: str
    ) -> None:
        """Close a position and record to Kelly calculator."""
        pos = self.positions[pos_idx]
        
        # Calculate P&L
        pnl = pos.compute_pnl(price)
        
        # Deduct fees
        fee_cost = pos.size * pos.entry_capital * self.fee
        pnl -= fee_cost
        
        # Update capital
        self.capital += pnl
        
        # Record trade for Kelly
        risk = abs(pos.entry_price - pos.stop_loss) * pos.size * pos.entry_capital
        self.kelly.add_trade(pnl, risk)
        
        # Log trade
        self.trade_log.append({
            "date": date,
            "action": "CLOSE",
            "strategy": pos.strategy,
            "direction": "LONG" if pos.direction == 1 else "SHORT",
            "entry_price": pos.entry_price,
            "exit_price": price,
            "size": pos.size,
            "pnl": pnl,
            "pnl_pct": (pnl / (pos.size * pos.entry_capital)) * 100,
            "reason": reason,
            "hold_days": (date - pos.entry_date).days,
            "fee": fee_cost,
            "trailing_used": pos.use_trailing_stop,
        })
    
    def close_all_positions(self, date: pd.Timestamp, price: float) -> None:
        """Close all open positions."""
        for i in reversed(range(len(self.positions))):
            self.close_position(i, date, price, "end_of_period")
        self.positions = []
    
    def get_stats(self) -> Dict:
        """Calculate performance statistics."""
        if not self.equity_curve:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] / self.initial_capital) - 1
        
        # Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Trades
        closed_trades = [t for t in self.trade_log if t["action"] == "CLOSE"]
        
        if closed_trades:
            pnls = [t["pnl"] for t in closed_trades]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            win_rate = len(winning_trades) / len(closed_trades)
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else 0
            
            # Trailing stop stats
            trailing_exits = [t for t in closed_trades if t.get("reason") == "trailing_stop"]
            trailing_pct = len(trailing_exits) / len(closed_trades) if closed_trades else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = trailing_pct = 0
        
        # Sharpe (annualized, assuming daily data)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
        else:
            sharpe = 0
        
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "total_trades": len(closed_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "trailing_stop_pct": trailing_pct,
            "final_capital": equity[-1],
        }


# =============================================================================
# WALK-FORWARD OPTIMIZATION
# =============================================================================

@dataclass
class StrategyParams:
    """Strategy parameter set for optimization."""
    trend_atr_mult: float = 2.0
    trend_trailing_activation: float = 1.5
    trend_trailing_distance: float = 1.0
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.02
    
    def to_dict(self) -> Dict:
        return asdict(self)


class WalkForwardOptimizer:
    """
    Walk-forward optimization:
    - Split data into training and testing windows
    - Optimize parameters on training data
    - Test on out-of-sample data
    - Roll forward and repeat
    """
    
    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 3,
        param_grid: Optional[Dict[str, List]] = None,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.param_grid = param_grid or self._default_param_grid()
    
    @staticmethod
    def _default_param_grid() -> Dict[str, List]:
        """Default parameter search grid."""
        return {
            "trend_atr_mult": [1.5, 2.0, 2.5],
            "trend_trailing_distance": [0.8, 1.0, 1.2],
            "kelly_fraction": [0.2, 0.25, 0.3],
            "max_risk_per_trade": [0.015, 0.02, 0.025],
        }
    
    def generate_param_combinations(self) -> List[StrategyParams]:
        """Generate all parameter combinations from grid."""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        params_list = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            params_list.append(StrategyParams(**param_dict))
        
        return params_list
    
    def optimize_window(
        self,
        df: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> StrategyParams:
        """
        Optimize parameters on training window.
        Returns best parameters based on Sharpe ratio.
        """
        param_combinations = self.generate_param_combinations()
        
        best_params = None
        best_sharpe = -np.inf
        
        train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        
        for params in param_combinations:
            try:
                # Run backtest with these params
                strategies = [
                    TrendFollowingStrategy(
                        atr_stop_mult=params.trend_atr_mult,
                        trailing_stop_activation=params.trend_trailing_activation,
                        trailing_stop_distance=params.trend_trailing_distance,
                    ),
                    MeanReversionStrategy(),
                    BreakoutStrategy(),
                ]
                
                portfolio, _ = backtest(
                    train_df,
                    strategies,
                    portfolio_kwargs={
                        "initial_capital": 10000,
                        "max_positions": 3,
                        "max_risk_per_trade": params.max_risk_per_trade,
                        "max_total_exposure": 1.0,
                        "fee_bps": 10.0,
                        "kelly_calculator": KellyCalculator(fraction=params.kelly_fraction),
                    }
                )
                
                stats = portfolio.get_stats()
                sharpe = stats.get("sharpe_ratio", -np.inf)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
            
            except Exception as e:
                continue
        
        return best_params or StrategyParams()
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        start_date: str,
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Run complete walk-forward optimization.
        
        Returns:
            results: List of results for each window
            combined_equity: Combined equity curve across all windows
        """
        df = df[df.index >= start_date].copy()
        
        results = []
        all_trades = []
        combined_equity = []
        
        current_date = df.index[0]
        end_date = df.index[-1]
        
        while current_date < end_date:
            # Define windows
            train_start = current_date
            train_end = current_date + pd.DateOffset(months=self.train_months)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_months)
            
            if test_end > end_date:
                test_end = end_date
            
            # Skip if not enough data
            if train_end >= end_date:
                break
            
            print(f"\nOptimizing: {train_start.date()} to {train_end.date()}")
            print(f"Testing: {test_start.date()} to {test_end.date()}")
            
            # Optimize on training data
            best_params = self.optimize_window(df, train_start, train_end)
            
            print(f"Best params: {best_params.to_dict()}")
            
            # Test on out-of-sample data
            test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
            
            strategies = [
                TrendFollowingStrategy(
                    atr_stop_mult=best_params.trend_atr_mult,
                    trailing_stop_activation=best_params.trend_trailing_activation,
                    trailing_stop_distance=best_params.trend_trailing_distance,
                ),
                MeanReversionStrategy(),
                BreakoutStrategy(),
            ]
            
            portfolio, _ = backtest(
                test_df,
                strategies,
                portfolio_kwargs={
                    "initial_capital": 10000,
                    "max_positions": 3,
                    "max_risk_per_trade": best_params.max_risk_per_trade,
                    "max_total_exposure": 1.0,
                    "fee_bps": 10.0,
                    "kelly_calculator": KellyCalculator(fraction=best_params.kelly_fraction),
                }
            )
            
            stats = portfolio.get_stats()
            
            results.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "params": best_params.to_dict(),
                "stats": stats,
            })
            
            all_trades.extend(portfolio.trade_log)
            combined_equity.extend(portfolio.equity_curve)
            
            # Move to next window
            current_date = test_end + pd.DateOffset(days=1)
        
        return results, pd.DataFrame({"equity": combined_equity})


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def backtest(
    df: pd.DataFrame,
    strategies: List,
    start_date: Optional[str] = None,
    portfolio_kwargs: Optional[Dict] = None,
) -> Tuple[Portfolio, pd.DataFrame]:
    """
    Run backtest with multiple strategies and regime detection.
    """
    if start_date:
        df = df[df.index >= start_date].copy()
    
    if portfolio_kwargs is None:
        portfolio_kwargs = {}
    
    portfolio = Portfolio(**portfolio_kwargs)
    all_signals = []
    
    for i in range(200, len(df)):
        date = df.index[i]
        row = df.iloc[i]
        price = row["Close"]
        
        # Detect regime
        regime = RegimeDetector.detect(row)
        
        # Update existing positions
        portfolio.update_positions(date, price)
        
        # Get Kelly size suggestion
        kelly_size = portfolio.kelly.calculate_kelly()
        
        # Generate signals from all strategies
        for strategy in strategies:
            signal = strategy.generate_signal(df, i, regime, kelly_size)
            
            if signal:
                all_signals.append({
                    "date": signal.date,
                    "strategy": signal.strategy,
                    "direction": "LONG" if signal.direction == 1 else "SHORT",
                    "confidence": signal.confidence,
                    "position_size": signal.position_size,
                    "entry": signal.entry_price,
                    "stop": signal.stop_loss,
                    "target": signal.take_profit,
                    "trailing": signal.use_trailing_stop,
                    "regime": regime,
                })
                
                # Try to open position
                portfolio.open_position(signal)
    
    # Close all positions at end
    portfolio.close_all_positions(df.index[-1], df.iloc[-1]["Close"])
    
    df_signals = pd.DataFrame(all_signals)
    
    return portfolio, df_signals


def print_backtest_results(portfolio: Portfolio, df: pd.DataFrame, start_idx: int = 200) -> None:
    """Print formatted backtest results."""
    stats = portfolio.get_stats()
    
    # Buy and hold comparison
    bh_return = (df.iloc[-1]["Close"] / df.iloc[start_idx]["Close"]) - 1
    period_start = df.index[start_idx]
    period_end = df.index[-1]
    strategy_lines: List[str] = []
    regime_lines: List[str] = []
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    
    print(f"\nPeriod: {period_start.date()} to {period_end.date()}")
    print(f"Initial Capital: ${portfolio.initial_capital:,.2f}")
    print(f"Final Capital: ${stats['final_capital']:,.2f}")
    
    print(f"\n{'PERFORMANCE METRICS':-^70}")
    print(f"Total Return: {stats['total_return']*100:>25.2f}%")
    print(f"Buy & Hold Return: {bh_return*100:>25.2f}%")
    print(f"Outperformance: {(stats['total_return']-bh_return)*100:>25.2f}%")
    print(f"Max Drawdown: {stats['max_drawdown']*100:>25.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:>25.2f}")
    
    print(f"\n{'TRADING METRICS':-^70}")
    print(f"Total Trades: {stats['total_trades']:>25}")
    print(f"Win Rate: {stats['win_rate']*100:>25.2f}%")
    print(f"Avg Win: ${stats['avg_win']:>25.2f}")
    print(f"Avg Loss: ${stats['avg_loss']:>25.2f}")
    print(f"Profit Factor: {stats['profit_factor']:>25.2f}")
    print(f"Trailing Stop Exits: {stats['trailing_stop_pct']*100:>25.2f}%")
    
    # Strategy breakdown
    closed_trades = [t for t in portfolio.trade_log if t["action"] == "CLOSE"]
    if closed_trades:
        print(f"\n{'STRATEGY BREAKDOWN':-^70}")
        df_trades = pd.DataFrame(closed_trades)
        
        for strategy in df_trades["strategy"].unique():
            strat_trades = df_trades[df_trades["strategy"] == strategy]
            strat_pnl = strat_trades["pnl"].sum()
            strat_win_rate = (strat_trades["pnl"] > 0).mean()
            line = (f"{strategy:>20}: {len(strat_trades):>4} trades | "
                    f"P&L: ${strat_pnl:>10.2f} | Win Rate: {strat_win_rate*100:>5.1f}%")
            strategy_lines.append(line.strip())
            print(line)
        
        # Regime breakdown
        print(f"\n{'REGIME BREAKDOWN':-^70}")
        if "regime" in df_trades.columns:
            for regime in df_trades["regime"].unique():
                reg_trades = df_trades[df_trades["regime"] == regime]
                reg_pnl = reg_trades["pnl"].sum()
                reg_win_rate = (reg_trades["pnl"] > 0).mean()
                line = (f"{regime:>20}: {len(reg_trades):>4} trades | "
                        f"P&L: ${reg_pnl:>10.2f} | Win Rate: {reg_win_rate*100:>5.1f}%")
                regime_lines.append(line.strip())
                print(line)
    
    print("\n" + "="*70)
    send_email_notification(
        stats,
        bh_return,
        period_start,
        period_end,
        portfolio.initial_capital,
        strategy_lines,
        regime_lines,
    )


# =============================================================================
# MAIN
# =============================================================================

def _format_stats_email(
    stats: Dict[str, Any],
    bh_return: float,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    initial_capital: float,
    strategy_lines: List[str],
    regime_lines: List[str],
) -> Tuple[str, str]:
    subject = f"BTC Analyzer Backtest Results {period_start.date()} to {period_end.date()}"

    lines = [
        f"Period: {period_start.date()} to {period_end.date()}",
        f"Initial Capital: ${initial_capital:,.2f}",
        f"Final Capital: ${stats['final_capital']:,.2f}",
        "",
        "PERFORMANCE METRICS",
        f"Total Return: {stats['total_return']*100:.2f}%",
        f"Buy & Hold Return: {bh_return*100:.2f}%",
        f"Outperformance: {(stats['total_return']-bh_return)*100:.2f}%",
        f"Max Drawdown: {stats['max_drawdown']*100:.2f}%",
        f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}",
        "",
        "TRADING METRICS",
        f"Total Trades: {stats['total_trades']}",
        f"Win Rate: {stats['win_rate']*100:.2f}%",
        f"Avg Win: ${stats['avg_win']:.2f}",
        f"Avg Loss: ${stats['avg_loss']:.2f}",
        f"Profit Factor: {stats['profit_factor']:.2f}",
        f"Trailing Stop Exits: {stats['trailing_stop_pct']*100:.2f}%",
    ]

    if strategy_lines:
        lines.extend(["", "STRATEGY BREAKDOWN", *strategy_lines])

    if regime_lines:
        lines.extend(["", "REGIME BREAKDOWN", *regime_lines])

    return subject, "\n".join(lines)


def send_email_notification(
    stats: Dict[str, Any],
    bh_return: float,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    initial_capital: float,
    strategy_lines: List[str],
    regime_lines: List[str],
) -> None:
    recipients = [e.strip() for e in os.getenv("BTC_NOTIFY_EMAILS", "").split(",") if e.strip()]
    if not recipients:
        return

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_password = os.getenv("SMTP_PASSWORD", "")
    sender = os.getenv("SMTP_SENDER", smtp_user)

    if not (smtp_host and smtp_user and smtp_password and sender):
        raise RuntimeError("Email enabled but SMTP settings are missing.")

    subject, body = _format_stats_email(
        stats,
        bh_return,
        period_start,
        period_end,
        initial_capital,
        strategy_lines,
        regime_lines,
    )
    message = f"Subject: {subject}\nFrom: {sender}\nTo: {', '.join(recipients)}\n\n{body}"

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, recipients, message)


def main():
    """Run the advanced trading system."""
    print("="*70)
    print("ADVANCED BITCOIN TRADING SYSTEM")
    print("="*70)
    
    print("\nFetching BTC data...")
    df = fetch_btc_data(start="2014-01-01")
    
    print("Adding indicators...")
    df = add_indicators(df)
    
    # Choose mode
    mode = os.getenv("MODE", "backtest")  # backtest or walk_forward
    
    if mode == "walk_forward":
        print("\n" + "="*70)
        print("WALK-FORWARD OPTIMIZATION MODE")
        print("="*70)
        
        optimizer = WalkForwardOptimizer(
            train_months=12,
            test_months=3,
        )
        
        results, combined_equity = optimizer.run_walk_forward(df, start_date="2016-01-01")
        
        # Print summary
        print("\n" + "="*70)
        print("WALK-FORWARD SUMMARY")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            stats = result["stats"]
            print(f"\nWindow {i}:")
            print(f"  Test Period: {result['test_start'].date()} to {result['test_end'].date()}")
            print(f"  Return: {stats['total_return']*100:.2f}%")
            print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
            print(f"  Trades: {stats['total_trades']}")
        
        # Save results
        with open("walk_forward_results.json", "w") as f:
            # Convert timestamps to strings for JSON serialization
            results_json = []
            for r in results:
                r_copy = r.copy()
                r_copy["train_start"] = r_copy["train_start"].isoformat()
                r_copy["train_end"] = r_copy["train_end"].isoformat()
                r_copy["test_start"] = r_copy["test_start"].isoformat()
                r_copy["test_end"] = r_copy["test_end"].isoformat()
                results_json.append(r_copy)
            
            json.dump(results_json, f, indent=2)
        
        combined_equity.to_csv("walk_forward_equity.csv", index=False)
        print("\nResults saved to walk_forward_results.json and walk_forward_equity.csv")
    
    else:
        print("\n" + "="*70)
        print("STANDARD BACKTEST MODE")
        print("="*70)
        
        print("\nInitializing strategies...")
        strategies = [
            TrendFollowingStrategy(
                atr_stop_mult=2.0,
                trailing_stop_activation=1.5,
                trailing_stop_distance=1.0,
            ),
            MeanReversionStrategy(),
            BreakoutStrategy(lookback=20),
        ]
        
        print("Running backtest...")
        portfolio, df_signals = backtest(
            df,
            strategies,
            start_date="2016-01-01",
            portfolio_kwargs={
                "initial_capital": 10000,
                "max_positions": 3,
                "max_risk_per_trade": 0.02,
                "max_total_exposure": 1.0,
                "fee_bps": 10.0,
                "kelly_calculator": KellyCalculator(fraction=0.25),
            }
        )
        
        print_backtest_results(portfolio, df)
        
        # Save results
        df_trades = pd.DataFrame(portfolio.trade_log)
        df_trades.to_csv("trades.csv", index=False)
        
        df_equity = pd.DataFrame({
            "date": df.index[200:200+len(portfolio.equity_curve)],
            "equity": portfolio.equity_curve,
        })
        df_equity.to_csv("equity_curve.csv", index=False)
        
        if not df_signals.empty:
            df_signals.to_csv("signals.csv", index=False)
        
        print("\nResults saved to trades.csv, equity_curve.csv, and signals.csv")
        
        # Show recent signals
        if not df_signals.empty:
            print(f"\n{'RECENT SIGNALS':-^70}")
            print(df_signals.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
