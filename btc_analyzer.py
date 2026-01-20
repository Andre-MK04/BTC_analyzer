from __future__ import annotations

import math
import json
import os
import smtplib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# pandas-ta provides many indicators quickly
from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator


# -----------------------------
# Data
# -----------------------------
def fetch_btc_usd_daily(
    start: str = "2014-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch BTC-USD daily OHLCV from Yahoo Finance (BTC-USD).
    This covers ~2014 onwards (good for 10-12 years of history).
    """
    ticker = "BTC-USD"
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data returned. Check your internet connection or ticker availability.")
    df = df.rename(columns=str.title)
    # Expect columns: Open, High, Low, Close, Adj Close, Volume
    # Keep a consistent set:
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def fetch_btc_usd_interval(
    interval: str,
    period: str,
) -> pd.DataFrame:
    ticker = "BTC-USD"
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for interval={interval}.")
    df = df.rename(columns=str.title)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df


def _ema_trend_from_close(close: pd.Series) -> str:
    close = close.dropna()
    if len(close) < 210:
        return "neutral"
    ema50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    ema200 = EMAIndicator(close, window=200).ema_indicator().iloc[-1]
    last_close = close.iloc[-1]
    if last_close > ema50 > ema200:
        return "up"
    if last_close < ema50 < ema200:
        return "down"
    return "neutral"


def get_mtf_trends() -> Dict[str, str]:
    data_1h = fetch_btc_usd_interval("1h", "730d")
    data_4h = fetch_btc_usd_interval("4h", "730d")
    data_1w = fetch_btc_usd_interval("1wk", "10y")
    return {
        "1h": _ema_trend_from_close(data_1h["Close"]),
        "4h": _ema_trend_from_close(data_4h["Close"]),
        "1w": _ema_trend_from_close(data_1w["Close"]),
    }


def _fit_logistic(scores: np.ndarray, outcomes: np.ndarray) -> Tuple[float, float]:
    a = 1.0
    b = 0.0
    lr = 0.05
    for _ in range(3000):
        z = a * scores + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_a = np.mean((p - outcomes) * scores)
        grad_b = np.mean(p - outcomes)
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)


def build_confidence_calibrator(
    df_ind: pd.DataFrame,
    atr_mult: float = 2.0,
    start_index: int = 250,
    max_samples: int = 1200,
) -> Tuple[float, float]:
    def _scalar(x):
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        return float(x)

    scores: List[float] = []
    outcomes: List[int] = []
    df = df_ind.dropna().copy()
    end_index = min(len(df) - 2, start_index + max_samples)
    for i in range(start_index, end_index):
        hist = df.iloc[: i + 1].copy()
        swings = zigzag_swings(hist["Close"], hist["ATR14"], atr_mult=atr_mult)
        sig = compute_signal_for_last_day(hist, swings, confidence_calibrator=None, mtf_trends=None)
        next_ret = _scalar(df["Close"].iloc[i + 1] / df["Close"].iloc[i] - 1.0)
        if sig.score == 0:
            continue
        correct = 1 if (sig.score > 0 and next_ret > 0) or (sig.score < 0 and next_ret < 0) else 0
        scores.append(abs(sig.score))
        outcomes.append(correct)

    if len(scores) < 50:
        return 1.0, 0.0
    return _fit_logistic(np.array(scores, dtype=float), np.array(outcomes, dtype=float))


# -----------------------------
# Indicators (broad set)
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = out["Close"].squeeze()
    high = out["High"].squeeze()
    low = out["Low"].squeeze()
    vol = out["Volume"].squeeze()

    # Trend
    out["EMA20"] = EMAIndicator(close, window=20).ema_indicator()
    out["EMA50"] = EMAIndicator(close, window=50).ema_indicator()
    out["EMA200"] = EMAIndicator(close, window=200).ema_indicator()
    out["SMA50"] = SMAIndicator(close, window=50).sma_indicator()
    out["SMA200"] = SMAIndicator(close, window=200).sma_indicator()

    adx = ADXIndicator(high, low, close, window=14)
    out["ADX_14"] = adx.adx()
    out["DMP_14"] = adx.adx_pos()
    out["DMN_14"] = adx.adx_neg()

    # Momentum
    out["RSI14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["MACD_12_26_9"] = macd.macd()
    out["MACDs_12_26_9"] = macd.macd_signal()
    out["MACDh_12_26_9"] = macd.macd_diff()

    out["ROC10"] = ROCIndicator(close, window=10).roc()

    st = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    out["STOCHRSIk_14_14_3_3"] = st.stochrsi_k()
    out["STOCHRSId_14_14_3_3"] = st.stochrsi_d()

    # Volatility
    out["ATR14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    bb = BollingerBands(close, window=20, window_dev=2)
    out["BBL_20_2.0"] = bb.bollinger_lband()
    out["BBM_20_2.0"] = bb.bollinger_mavg()
    out["BBU_20_2.0"] = bb.bollinger_hband()
    out["BBP_20_2.0"] = bb.bollinger_pband()
    # bandwidth approximation (not identical to pandas-ta, but good)
    out["BBB_20_2.0"] = (out["BBU_20_2.0"] - out["BBL_20_2.0"]) / (out["BBM_20_2.0"] + 1e-9)

    kc = KeltnerChannel(high, low, close, window=20, window_atr=10)
    out["KCL_20_2"] = kc.keltner_channel_lband()
    out["KCM_20_2"] = kc.keltner_channel_mband()
    out["KCU_20_2"] = kc.keltner_channel_hband()

    # Volume / flow
    out["OBV"] = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    out["MFI14"] = MFIIndicator(high, low, close, vol, window=14).money_flow_index()
    out["CMF20"] = ChaikinMoneyFlowIndicator(high, low, close, vol, window=20).chaikin_money_flow()

    # Regime helpers
    out["Return1D"] = out["Close"].pct_change()
    out["Volatility20"] = out["Return1D"].rolling(20).std() * math.sqrt(365)

    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


# -----------------------------
# ZigZag swing detection
# -----------------------------
@dataclass
class SwingPoint:
    idx: pd.Timestamp
    price: float
    kind: str  # "H" or "L"


def zigzag_swings(
    close: pd.Series,
    atr: pd.Series,
    atr_mult: float = 2.0,
    min_bars: int = 3,
) -> List[SwingPoint]:
    """
    ATR-based ZigZag on Close.
    A swing is confirmed when price reverses by atr_mult * ATR from last pivot.
    """
    close = close.dropna()
    atr = atr.reindex(close.index).ffill().bfill()

    if close.empty:
        return []

    def _scalar(x):
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        return float(x)

    swings: List[SwingPoint] = []
    last_pivot_idx = close.index[0]
    last_pivot_price = _scalar(close.iloc[0])
    trend = 0  # 1 up, -1 down, 0 unknown

    candidate_extreme_idx = last_pivot_idx
    candidate_extreme_price = last_pivot_price

    for i in range(1, len(close)):
        idx = close.index[i]
        price = _scalar(close.iloc[i])
        threshold = _scalar(atr.iloc[i] * atr_mult)

        if trend >= 0:
            # tracking highs
            if price >= candidate_extreme_price:
                candidate_extreme_price = price
                candidate_extreme_idx = idx

            # reversal down enough?
            if candidate_extreme_price - price >= threshold:
                # confirm HIGH pivot
                if len(swings) == 0 or (swings[-1].kind != "H"):
                    swings.append(SwingPoint(candidate_extreme_idx, candidate_extreme_price, "H"))
                trend = -1
                last_pivot_idx = candidate_extreme_idx
                last_pivot_price = candidate_extreme_price
                candidate_extreme_idx = idx
                candidate_extreme_price = price

        if trend <= 0:
            # tracking lows
            if price <= candidate_extreme_price:
                candidate_extreme_price = price
                candidate_extreme_idx = idx

            # reversal up enough?
            if price - candidate_extreme_price >= threshold:
                # confirm LOW pivot
                if len(swings) == 0 or (swings[-1].kind != "L"):
                    swings.append(SwingPoint(candidate_extreme_idx, candidate_extreme_price, "L"))
                trend = 1
                last_pivot_idx = candidate_extreme_idx
                last_pivot_price = candidate_extreme_price
                candidate_extreme_idx = idx
                candidate_extreme_price = price

    # basic cleanup: remove tiny oscillations too close in time
    cleaned: List[SwingPoint] = []
    for sp in swings:
        if not cleaned:
            cleaned.append(sp)
            continue
        if (sp.idx - cleaned[-1].idx).days < min_bars:
            # keep the more extreme of same kind
            if sp.kind == cleaned[-1].kind:
                if sp.kind == "H" and sp.price > cleaned[-1].price:
                    cleaned[-1] = sp
                elif sp.kind == "L" and sp.price < cleaned[-1].price:
                    cleaned[-1] = sp
            else:
                # too close but alternating; keep both (rare)
                cleaned.append(sp)
        else:
            cleaned.append(sp)
    return cleaned


# -----------------------------
# Harmonics (Butterfly framework)
# -----------------------------
def _ratio(a: float, b: float) -> float:
    return abs(a / b) if b != 0 else np.nan


@dataclass
class HarmonicPattern:
    name: str
    direction: str  # "bullish" (D is low) or "bearish" (D is high)
    points: Dict[str, Tuple[pd.Timestamp, float]]  # X,A,B,C,D
    quality: float  # 0..1


def find_butterfly_patterns(
    swings: List[SwingPoint],
    tol: float = 0.08,
) -> List[HarmonicPattern]:
    """
    Detect Butterfly patterns using XABCD from swings.

    Common butterfly guidelines (approx):
    - B retrace of XA: ~0.786
    - C retrace of AB: 0.382..0.886
    - D extension of XA: 1.27..1.618
    We'll compute "quality" based on closeness to ideal ratios.
    """
    patterns: List[HarmonicPattern] = []
    if len(swings) < 5:
        return patterns

    # Build sequences of 5 alternating pivots
    for i in range(len(swings) - 4):
        seq = swings[i : i + 5]
        # Require alternating kinds (H/L/H/L/H or L/H/L/H/L)
        kinds = [s.kind for s in seq]
        if any(kinds[j] == kinds[j + 1] for j in range(4)):
            continue

        X, A, B, C, D = seq

        XA = A.price - X.price
        AB = B.price - A.price
        BC = C.price - B.price
        CD = D.price - C.price

        # Determine bullish/bearish by overall shape:
        # Bullish butterfly typically ends with D as a low (last pivot is L)
        direction = "bullish" if D.kind == "L" else "bearish"

        # Ratios:
        # B retracement of XA
        rB = _ratio(B.price - A.price, X.price - A.price) if XA != 0 else np.nan
        # Use absolute retracement: |AB| / |XA|
        rB = abs(AB) / abs(XA) if XA != 0 else np.nan

        # C retracement of AB
        rC = abs(BC) / abs(AB) if AB != 0 else np.nan

        # D extension of XA (|AD| / |XA|)
        AD = D.price - A.price
        rD = abs(AD) / abs(XA) if XA != 0 else np.nan

        # Rules
        okB = abs(rB - 0.786) <= tol
        okC = (0.382 - tol) <= rC <= (0.886 + tol)
        okD = (1.27 - tol) <= rD <= (1.618 + tol)

        if not (okB and okC and okD):
            continue

        # Quality score: closeness to key targets
        qB = max(0.0, 1.0 - abs(rB - 0.786) / tol) if not np.isnan(rB) else 0.0
        # For C, closeness to mid-range (0.618) rather than edges
        qC = max(0.0, 1.0 - abs(rC - 0.618) / (0.5 + tol)) if not np.isnan(rC) else 0.0
        # For D, closeness to 1.414 midpoint
        qD = max(0.0, 1.0 - abs(rD - 1.414) / (0.35 + tol)) if not np.isnan(rD) else 0.0
        quality = float(np.clip((qB + qC + qD) / 3.0, 0.0, 1.0))

        patterns.append(
            HarmonicPattern(
                name="Butterfly",
                direction=direction,
                points={
                    "X": (X.idx, X.price),
                    "A": (A.idx, A.price),
                    "B": (B.idx, B.price),
                    "C": (C.idx, C.price),
                    "D": (D.idx, D.price),
                },
                quality=quality,
            )
        )

    return patterns


# -----------------------------
# Elliott heuristic (lightweight)
# -----------------------------
@dataclass
class ElliottSignal:
    impulse_prob: float  # 0..1
    correction_prob: float  # 0..1
    note: str


def elliott_impulse_vs_correction(swings: List[SwingPoint]) -> ElliottSignal:
    """
    Simple heuristic:
    - If last 6-8 swings show trending structure (HH/HL or LL/LH) and wave-3-like extension,
      raise impulse_prob.
    - If swings are overlapping and mean-reverting, raise correction_prob.
    """
    if len(swings) < 6:
        return ElliottSignal(0.5, 0.5, "Not enough swings")

    recent = swings[-8:] if len(swings) >= 8 else swings
    prices = np.array([s.price for s in recent], dtype=float)
    kinds = [s.kind for s in recent]

    # overlap measure: how often ranges overlap between successive legs
    legs = np.abs(np.diff(prices))
    leg_mean = float(np.mean(legs)) if len(legs) else 0.0
    leg_std = float(np.std(legs)) if len(legs) else 0.0

    # trend structure
    highs = [s.price for s in recent if s.kind == "H"]
    lows = [s.price for s in recent if s.kind == "L"]

    hh = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
    hl = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])
    ll = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i - 1])
    lh = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i - 1])

    uptrend_score = (hh + hl) / max(1, (len(highs) - 1 + len(lows) - 1))
    downtrend_score = (ll + lh) / max(1, (len(highs) - 1 + len(lows) - 1))
    trendiness = max(uptrend_score, downtrend_score)

    # wave-3-ish extension: one leg much bigger than others
    extension = 0.0
    if len(legs) >= 3 and leg_mean > 0:
        extension = float(np.max(legs) / leg_mean)

    impulse = 0.35 * trendiness + 0.25 * np.clip((extension - 1.2) / 1.8, 0, 1) + 0.20 * np.clip(leg_std / (leg_mean + 1e-9), 0, 1)
    impulse = float(np.clip(impulse, 0, 1))
    correction = float(np.clip(1.0 - impulse * 0.9, 0, 1))

    note = f"trendiness={trendiness:.2f}, extension={extension:.2f}"
    return ElliottSignal(impulse, correction, note)


# -----------------------------
# Chart patterns (rule-based)
# -----------------------------
@dataclass
class ChartPattern:
    name: str
    direction: str  # bullish/bearish
    idx: pd.Timestamp
    strength: float  # 0..1
    meta: Dict[str, float]


def _near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def detect_double_top_bottom(swings: List[SwingPoint], pct_tol: float = 0.03) -> List[ChartPattern]:
    """
    Double top: H - L - H with highs near each other and middle low lower.
    Double bottom: L - H - L similarly.
    """
    out: List[ChartPattern] = []
    if len(swings) < 3:
        return out

    for i in range(len(swings) - 2):
        a, b, c = swings[i], swings[i + 1], swings[i + 2]
        if a.kind == "H" and b.kind == "L" and c.kind == "H":
            tol = pct_tol * ((a.price + c.price) / 2)
            if _near(a.price, c.price, tol):
                strength = float(np.clip(1.0 - abs(a.price - c.price) / (tol + 1e-9), 0, 1))
                out.append(ChartPattern("Double Top", "bearish", c.idx, strength, {"a": a.price, "c": c.price, "b": b.price}))
        if a.kind == "L" and b.kind == "H" and c.kind == "L":
            tol = pct_tol * ((a.price + c.price) / 2)
            if _near(a.price, c.price, tol):
                strength = float(np.clip(1.0 - abs(a.price - c.price) / (tol + 1e-9), 0, 1))
                out.append(ChartPattern("Double Bottom", "bullish", c.idx, strength, {"a": a.price, "c": c.price, "b": b.price}))
    return out


def detect_head_and_shoulders(swings: List[SwingPoint], pct_tol: float = 0.05) -> List[ChartPattern]:
    """
    H&S: H L H L H where middle H is highest, shoulders near each other.
    Inverse H&S: L H L H L where middle L is lowest.
    """
    out: List[ChartPattern] = []
    if len(swings) < 5:
        return out

    for i in range(len(swings) - 4):
        s = swings[i : i + 5]
        kinds = [x.kind for x in s]
        if any(kinds[j] == kinds[j + 1] for j in range(4)):
            continue

        p = [x.price for x in s]
        # Normal H&S
        if kinds == ["H", "L", "H", "L", "H"]:
            left_sh, left_low, head, right_low, right_sh = s
            if head.price > left_sh.price and head.price > right_sh.price:
                shoulder_tol = pct_tol * ((left_sh.price + right_sh.price) / 2)
                if _near(left_sh.price, right_sh.price, shoulder_tol):
                    neckline = (left_low.price + right_low.price) / 2
                    strength = float(np.clip(
                        (head.price - neckline) / (head.price + 1e-9), 0, 1
                    ))
                    out.append(ChartPattern("Head & Shoulders", "bearish", right_sh.idx, strength, {
                        "left_sh": left_sh.price, "head": head.price, "right_sh": right_sh.price, "neckline": neckline
                    }))

        # Inverse H&S
        if kinds == ["L", "H", "L", "H", "L"]:
            left_sh, left_high, head, right_high, right_sh = s
            if head.price < left_sh.price and head.price < right_sh.price:
                shoulder_tol = pct_tol * ((left_sh.price + right_sh.price) / 2)
                if _near(left_sh.price, right_sh.price, shoulder_tol):
                    neckline = (left_high.price + right_high.price) / 2
                    strength = float(np.clip(
                        (neckline - head.price) / (neckline + 1e-9), 0, 1
                    ))
                    out.append(ChartPattern("Inverse Head & Shoulders", "bullish", right_sh.idx, strength, {
                        "left_sh": left_sh.price, "head": head.price, "right_sh": right_sh.price, "neckline": neckline
                    }))
    return out


# -----------------------------
# Signal scoring (ensemble)
# -----------------------------
@dataclass
class SignalOutput:
    date: pd.Timestamp
    action: str  # LONG/SHORT/NO TRADE
    confidence: float  # 0..1
    score: float
    reasons: List[str]
    levels: Dict[str, float]
    close: float
    atr: float
    atr_pct: float
    threshold: float
    score_breakdown: Dict[str, float]
    last_pattern_date: Optional[pd.Timestamp]


def regime_filter(row: pd.Series) -> str:
    """
    Simple regime:
    - trending if ADX high and BB bandwidth not tiny.
    - ranging otherwise.
    """
    def _scalar(x):
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        return float(x)

    adx = _scalar(row.get("ADX_14", np.nan))
    bbbw = _scalar(row.get("BBB_20_2.0", np.nan))  # BB bandwidth (pandas-ta)
    if np.isnan(adx) or np.isnan(bbbw):
        return "unknown"
    if adx >= 22 and bbbw >= 0.05:
        return "trending"
    return "ranging"


def compute_signal_for_last_day(
    df_ind: pd.DataFrame,
    swings: List[SwingPoint],
    confidence_calibrator: Optional[Tuple[float, float]] = None,
    mtf_trends: Optional[Dict[str, str]] = None,
    atr_pct_min: Optional[float] = None,
    atr_pct_max: Optional[float] = None,
    fib_tp_wiggle: float = 0.98,
    fib_stop_wiggle: float = 0.99,
) -> SignalOutput:
    last = df_ind.iloc[-1]
    date = df_ind.index[-1]
    def _scalar(x):
        # Convert possible 1-element Series/ndarray into a python float
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        return float(x)

    close = _scalar(last["Close"])
    ema50 = _scalar(last["EMA50"])
    ema200 = _scalar(last["EMA200"])

    reasons: List[str] = []
    score = 0.0
    breakdown: Dict[str, float] = {}

    # Trend score
    trend_score = 0.0
    if close > ema50 > ema200:
        trend_score = 0.6
        reasons.append("Uptrend (Close > EMA50 > EMA200)")
    elif close < ema50 < ema200:
        trend_score = -0.6
        reasons.append("Downtrend (Close < EMA50 < EMA200)")
    score += trend_score
    breakdown["trend"] = trend_score

    # Momentum score
    rsi_score = 0.0
    rsi = _scalar(last.get("RSI14", np.nan))
    if not np.isnan(rsi):
        if rsi <= 30:
            rsi_score = 0.25
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 70:
            rsi_score = -0.25
            reasons.append(f"RSI overbought ({rsi:.1f})")
    score += rsi_score
    breakdown["rsi"] = rsi_score

    macd_score = 0.0
    macdh = _scalar(last.get("MACDh_12_26_9", np.nan))
    if not np.isnan(macdh):
        macd_score = 0.15 if macdh > 0 else -0.15
        reasons.append(f"MACD histogram {'positive' if macdh>0 else 'negative'}")
    score += macd_score
    breakdown["macd_hist"] = macd_score

    # Regime
    reg = regime_filter(last)
    reasons.append(f"Regime: {reg}")

    # Elliott
    e = elliott_impulse_vs_correction(swings)
    elliott_score = (e.impulse_prob - 0.5) * 0.35
    score += elliott_score
    breakdown["elliott"] = elliott_score
    reasons.append(f"Elliott heuristic: impulse={e.impulse_prob:.2f} ({e.note})")

    # Harmonics (use most recent completed butterfly if near end)
    butterflies = find_butterfly_patterns(swings, tol=0.08)
    recent_bfly = None
    bfly_score = 0.0
    last_bfly_date = None
    if butterflies:
        # pick the last by D date
        recent_bfly = sorted(butterflies, key=lambda p: p.points["D"][0])[-1]
        d_date, d_price = recent_bfly.points["D"]
        last_bfly_date = d_date
        # if D is within last ~45 days, consider it relevant
        if (date - d_date).days <= 45:
            dist = abs(close - d_price) / close
            if dist <= 0.06:  # within 6% of PRZ point
                boost = 0.45 * recent_bfly.quality
                if recent_bfly.direction == "bullish":
                    bfly_score += boost
                    reasons.append(f"Bullish Butterfly nearby (q={recent_bfly.quality:.2f})")
                else:
                    bfly_score -= boost
                    reasons.append(f"Bearish Butterfly nearby (q={recent_bfly.quality:.2f})")
    score += bfly_score
    breakdown["butterfly"] = bfly_score

    # Chart patterns (recent)
    patterns = []
    patterns += detect_double_top_bottom(swings, pct_tol=0.03)
    patterns += detect_head_and_shoulders(swings, pct_tol=0.05)
    chart_score = 0.0
    last_chart_date = None
    if patterns:
        recent = sorted(patterns, key=lambda p: p.idx)[-1]
        last_chart_date = recent.idx
        if (date - recent.idx).days <= 60:
            adj = 0.35 * recent.strength
            chart_score += adj if recent.direction == "bullish" else -adj
            reasons.append(f"{recent.name} detected (strength={recent.strength:.2f})")
    score += chart_score
    breakdown["chart_patterns"] = chart_score

    # Volatility / risk-adjust
    atr = _scalar(last.get("ATR14", np.nan))
    levels = {}
    direction = "LONG" if score > 0 else "SHORT"
    if not np.isnan(atr):
        levels["stop_atr_1x"] = close - atr if direction == "LONG" else close + atr
        levels["tp_atr_2x"] = close + 2 * atr if direction == "LONG" else close - 2 * atr

    # Decide action
    T = 0.45 if reg == "trending" else 0.55  # stricter in ranging markets
    if score > T:
        action = "LONG"
    elif score < -T:
        action = "SHORT"
    else:
        action = "NO TRADE"

    atr_pct = (atr / close * 100.0) if not np.isnan(atr) and close != 0 else np.nan
    if not np.isnan(atr_pct):
        if atr_pct_min is not None and atr_pct < atr_pct_min:
            reasons.append(f"ATR% too low ({atr_pct:.2f}%) -> NO TRADE")
            action = "NO TRADE"
        if atr_pct_max is not None and atr_pct > atr_pct_max:
            reasons.append(f"ATR% too high ({atr_pct:.2f}%) -> NO TRADE")
            action = "NO TRADE"

    if mtf_trends and action in ("LONG", "SHORT"):
        expected = "up" if action == "LONG" else "down"
        mismatched = [k for k, v in mtf_trends.items() if v != expected]
        if mismatched:
            reasons.append(f"MTF trend mismatch ({', '.join(mismatched)}) -> NO TRADE")
            action = "NO TRADE"

    if confidence_calibrator:
        a, b = confidence_calibrator
        confidence = float(1.0 / (1.0 + np.exp(-(a * abs(score) + b))))
    else:
        confidence = float(np.clip(abs(score) / 1.2, 0, 1))
    last_pattern_date = None
    if last_bfly_date and last_chart_date:
        last_pattern_date = max(last_bfly_date, last_chart_date)
    elif last_bfly_date:
        last_pattern_date = last_bfly_date
    elif last_chart_date:
        last_pattern_date = last_chart_date

    if action in ("LONG", "SHORT") and swings:
        recent_swings = swings[-6:] if len(swings) >= 6 else swings
        highs = [s.price for s in recent_swings if s.kind == "H"]
        lows = [s.price for s in recent_swings if s.kind == "L"]
        if highs and lows:
            swing_high = max(highs)
            swing_low = min(lows)
            swing_range = swing_high - swing_low
            if swing_range > 0:
                if action == "LONG":
                    fib_stop = swing_low - 0.236 * swing_range
                    fib_tp = swing_low + 1.272 * swing_range
                    levels["stop_fib"] = fib_stop * fib_stop_wiggle
                    levels["tp_fib"] = fib_tp * fib_tp_wiggle
                else:
                    fib_stop = swing_high + 0.236 * swing_range
                    fib_tp = swing_high - 1.272 * swing_range
                    levels["stop_fib"] = fib_stop * (2.0 - fib_stop_wiggle)
                    levels["tp_fib"] = fib_tp * (2.0 - fib_tp_wiggle)

        ma_candidates = [ema50, ema200]
        if action == "LONG":
            ma_stop = min(ma_candidates)
            ma_tp_candidates = [ma for ma in ma_candidates if ma > close]
            ma_tp = min(ma_tp_candidates) if ma_tp_candidates else None
        else:
            ma_stop = max(ma_candidates)
            ma_tp_candidates = [ma for ma in ma_candidates if ma < close]
            ma_tp = max(ma_tp_candidates) if ma_tp_candidates else None
        if not np.isnan(ma_stop):
            levels["stop_ma"] = float(ma_stop)
        if ma_tp is not None:
            levels["tp_ma"] = float(ma_tp)
    return SignalOutput(
        date=date,
        action=action,
        confidence=confidence,
        score=float(score),
        reasons=reasons,
        levels=levels,
        close=float(close),
        atr=float(atr) if not np.isnan(atr) else np.nan,
        atr_pct=float(atr_pct) if not np.isnan(atr_pct) else np.nan,
        threshold=float(T),
        score_breakdown=breakdown,
        last_pattern_date=last_pattern_date,
    )


def _format_signal_email(
    sig: SignalOutput,
    extra_sections: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, str]:
    subject = f"BTC Signal for {sig.date.date()}: {sig.action} (conf {sig.confidence:.2f})"
    lines = [
        "BTC next-day signal",
        f"Date:       {sig.date.date()}",
        f"Close:      {sig.close:.2f}",
        f"Action:     {sig.action}",
        f"Score:      {sig.score:.3f}",
        f"Threshold:  {sig.threshold:.2f}",
        f"Confidence: {sig.confidence:.2f}",
    ]
    if not np.isnan(sig.atr):
        atr_pct = f"{sig.atr_pct:.2f}%" if not np.isnan(sig.atr_pct) else "n/a"
        lines.append(f"ATR14:      {sig.atr:.2f} ({atr_pct})")
    if sig.levels:
        lines.append("Levels:")
        for k, v in sig.levels.items():
            lines.append(f"  - {k}: {v:.2f}")
    if sig.score_breakdown:
        lines.append("Score breakdown:")
        for k, v in sig.score_breakdown.items():
            lines.append(f"  - {k}: {v:+.3f}")
    if sig.last_pattern_date:
        days_ago = (sig.date - sig.last_pattern_date).days
        lines.append(f"Last pattern: {sig.last_pattern_date.date()} ({days_ago} days ago)")
    else:
        lines.append("Last pattern: none found")
    if extra_sections:
        for title, section_lines in extra_sections.items():
            if not section_lines:
                continue
            lines.append(f"{title}:")
            for line in section_lines:
                lines.append(f"  - {line}")
    if sig.reasons:
        lines.append("Reasons:")
        for r in sig.reasons:
            lines.append(f"  - {r}")
    body = "\n".join(lines)
    return subject, body


def send_email_notification(
    sig: SignalOutput,
    extra_sections: Optional[Dict[str, List[str]]] = None,
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

    subject, body = _format_signal_email(sig, extra_sections=extra_sections)
    message = f"Subject: {subject}\nFrom: {sender}\nTo: {', '.join(recipients)}\n\n{body}"

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, recipients, message)


# -----------------------------
# Walk-forward backtest (baseline)
# -----------------------------
def walk_forward_backtest(
    df_ind: pd.DataFrame,
    atr_mult: float = 2.0,
    start_year: int = 2016,
    fee_bps: float = 10.0,  # 10 bps = 0.10% round-trip rough placeholder
    atr_pct_min: Optional[float] = None,
    atr_pct_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Baseline backtest:
    - each day, compute swings using all data up to that day
    - compute signal for that day close -> take position next day close-to-close
    - includes simple fee model
    """
    df = df_ind.copy().dropna()
    df = df[df.index.year >= start_year].copy()
    if len(df) < 300:
        raise RuntimeError("Not enough data after filters for backtest.")

    results = []
    fee = fee_bps / 10000.0
    def _scalar(x):
        if isinstance(x, pd.Series):
            return float(x.iloc[0])
        if isinstance(x, np.ndarray):
            return float(x.reshape(-1)[0])
        return float(x)

    for i in range(250, len(df) - 1):
        hist = df.iloc[: i + 1].copy()
        swings = zigzag_swings(hist["Close"], hist["ATR14"], atr_mult=atr_mult)
        sig = compute_signal_for_last_day(
            hist,
            swings,
            atr_pct_min=atr_pct_min,
            atr_pct_max=atr_pct_max,
        )

        next_ret = _scalar(df["Close"].iloc[i + 1] / df["Close"].iloc[i] - 1.0)
        pos = 0
        if sig.action == "LONG":
            pos = 1
        elif sig.action == "SHORT":
            pos = -1

        strat_ret = pos * next_ret
        # fee if you trade
        if pos != 0:
            strat_ret -= fee

        results.append({
            "Date": df.index[i],
            "Action": sig.action,
            "Score": sig.score,
            "Confidence": sig.confidence,
            "NextDayReturn": next_ret,
            "StrategyReturn": strat_ret,
        })

    res = pd.DataFrame(results).set_index("Date")
    res["Equity"] = (1.0 + res["StrategyReturn"]).cumprod()
    return res


def optimize_atr_params(
    df_ind: pd.DataFrame,
    atr_mult_grid: List[float],
    atr_pct_min_grid: List[Optional[float]],
    atr_pct_max_grid: List[Optional[float]],
    start_year: int = 2016,
    fee_bps: float = 10.0,
) -> pd.DataFrame:
    rows = []
    for atr_mult in atr_mult_grid:
        for atr_pct_min in atr_pct_min_grid:
            for atr_pct_max in atr_pct_max_grid:
                if atr_pct_min is not None and atr_pct_max is not None and atr_pct_min >= atr_pct_max:
                    continue
                bt = walk_forward_backtest(
                    df_ind,
                    atr_mult=atr_mult,
                    start_year=start_year,
                    fee_bps=fee_bps,
                    atr_pct_min=atr_pct_min,
                    atr_pct_max=atr_pct_max,
                )
                total_return = float(bt["Equity"].iloc[-1] - 1.0)
                max_dd = float((bt["Equity"] / bt["Equity"].cummax() - 1.0).min())
                hit_rate = float((bt["StrategyReturn"] > 0).mean())
                trades = int((bt["Action"] != "NO TRADE").sum())
                rows.append({
                    "atr_mult": atr_mult,
                    "atr_pct_min": atr_pct_min,
                    "atr_pct_max": atr_pct_max,
                    "total_return": total_return,
                    "max_drawdown": max_dd,
                    "hit_rate": hit_rate,
                    "trades": trades,
                })
    res = pd.DataFrame(rows)
    return res.sort_values(["total_return", "hit_rate"], ascending=[False, False]).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    df = fetch_btc_usd_daily(start="2014-01-01")
    df_ind = add_indicators(df)

    # swings on full history for pattern mining
    swings = zigzag_swings(df_ind["Close"], df_ind["ATR14"], atr_mult=2.0)
    confidence_calibrator = build_confidence_calibrator(df_ind, atr_mult=2.0)
    try:
        mtf_trends = get_mtf_trends()
    except Exception:
        mtf_trends = None
    sig = compute_signal_for_last_day(
        df_ind,
        swings,
        confidence_calibrator=confidence_calibrator,
        mtf_trends=mtf_trends,
    )
    send_email_notification(sig)

    print("\n=== NEXT-DAY SIGNAL (research) ===")
    print(f"Date:       {sig.date.date()}")
    print(f"Action:     {sig.action}")
    print(f"Score:      {sig.score:.3f}")
    print(f"Confidence: {sig.confidence:.2f}")
    if sig.levels:
        print("Levels:")
        for k, v in sig.levels.items():
            print(f"  - {k}: {v:.2f}")

    print("\nReasons:")
    for r in sig.reasons:
        print(f"  - {r}")

    # Pattern summary over 10-12 years
    butterflies = find_butterfly_patterns(swings, tol=0.08)
    patterns = []
    patterns += detect_double_top_bottom(swings, pct_tol=0.03)
    patterns += detect_head_and_shoulders(swings, pct_tol=0.05)

    print("\n=== PATTERNS FOUND (full history) ===")
    print(f"Butterflies: {len(butterflies)}")
    if butterflies:
        last_b = sorted(butterflies, key=lambda p: p.points['D'][0])[-1]
        print(f"  Last Butterfly: {last_b.direction}, q={last_b.quality:.2f}, D={last_b.points['D'][0].date()} @ {last_b.points['D'][1]:.2f}")

    print(f"Chart patterns: {len(patterns)}")
    if patterns:
        last_p = sorted(patterns, key=lambda p: p.idx)[-1]
        print(f"  Last Pattern: {last_p.name} ({last_p.direction}), strength={last_p.strength:.2f}, date={last_p.idx.date()}")

    # Backtest
    print("\n=== WALK-FORWARD BACKTEST (baseline) ===")
    bt = walk_forward_backtest(df_ind, atr_mult=2.0, start_year=2016, fee_bps=10.0)
    total_return = float(bt["Equity"].iloc[-1] - 1.0)
    max_dd = float((bt["Equity"] / bt["Equity"].cummax() - 1.0).min())
    hit_rate = float((bt["StrategyReturn"] > 0).mean())

    print(f"Trades:      {int((bt['Action'] != 'NO TRADE').sum())}")
    print(f"Hit rate:    {hit_rate:.2%}")
    print(f"Total return:{total_return:.2%}")
    print(f"Max drawdown:{max_dd:.2%}")
    print(f"Equity last: {bt['Equity'].iloc[-1]:.3f}")

    # Optional: save results
    bt.to_csv("backtest_results.csv")
    print("\nSaved: backtest_results.csv")

    if os.getenv("OPTIMIZE_ATR", ""):
        print("\n=== OPTIMIZE ATR PARAMS ===")
        atr_mult_grid = [round(x, 2) for x in np.arange(1.0, 3.25, 0.25)]
        atr_pct_min_grid = [None, 1.0, 1.5, 2.0]
        atr_pct_max_grid = [None, 8.0, 10.0, 12.0]
        opt = optimize_atr_params(
            df_ind,
            atr_mult_grid=atr_mult_grid,
            atr_pct_min_grid=atr_pct_min_grid,
            atr_pct_max_grid=atr_pct_max_grid,
            start_year=2016,
            fee_bps=10.0,
        )
        print(opt.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
