import pandas as pd
import numpy as np

from btc_analyzer import (
    fetch_btc_usd_daily,
    add_indicators,
    zigzag_swings,
    compute_signal_for_last_day,
    build_confidence_calibrator,
    get_mtf_trends,
    send_email_notification,
)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"

def fmt_money(x: float) -> str:
    return f"{x:.4f}"


def forward_simulate(
    start_date="2025-01-01",
    risk_per_trade=0.01,
    confidence_threshold=0.5,
    fee=0.001,  # 0.1%
    max_hold_days=5,
    send_email: bool = False,
):
    df_full = fetch_btc_usd_daily(start="2014-01-01")
    df_full = add_indicators(df_full)
    df = df_full[df_full.index >= start_date].copy()
    calibrator_df = df_full[df_full.index < start_date]
    confidence_calibrator = build_confidence_calibrator(
        calibrator_df if len(calibrator_df) >= 400 else df_full
    )

    capital = 1.0
    open_trade = None

    equity_curve = []
    trade_count = 0
    win_count = 0
    last_trade_pnl = None

    for i in range(len(df) - 1):
        today_hist = df.iloc[: i + 1]
        tomorrow_row = df.iloc[i + 1]

        # Compute today's signal (based only on history up to today)
        swings = zigzag_swings(today_hist["Close"], today_hist["ATR14"])
        sig = compute_signal_for_last_day(
            today_hist,
            swings,
            confidence_calibrator=confidence_calibrator,
        )

        # --- Manage open trade on "tomorrow close" (simple model) ---
        realized_today = 0.0
        closed_today = False

        if open_trade is not None:
            open_trade["days"] += 1
            close_price = float(tomorrow_row["Close"])
            high_price = float(tomorrow_row["High"])
            low_price = float(tomorrow_row["Low"])

            def close_trade(exit_price: float):
                nonlocal capital, trade_count, win_count, last_trade_pnl, realized_today, closed_today, open_trade
                entry = open_trade["entry"]
                direction = open_trade["direction"]

                if direction == "LONG":
                    pnl = (exit_price / entry - 1.0)
                else:  # SHORT
                    pnl = (entry / exit_price - 1.0)

                # Apply pnl scaled by risk_per_trade
                realized_today = pnl * open_trade["risk"]
                capital *= (1.0 + realized_today)

                # Fees on exit
                capital *= (1.0 - fee)

                trade_count += 1
                if pnl > 0:
                    win_count += 1
                last_trade_pnl = pnl
                closed_today = True
                open_trade = None

            # Stop/Target checks (use next-day high/low)
            if open_trade["direction"] == "LONG":
                if low_price <= open_trade["stop"]:
                    close_trade(float(open_trade["stop"]))
                elif high_price >= open_trade["target"]:
                    close_trade(float(open_trade["target"]))
            else:  # SHORT
                if high_price >= open_trade["stop"]:
                    close_trade(float(open_trade["stop"]))
                elif low_price <= open_trade["target"]:
                    close_trade(float(open_trade["target"]))

            # Time stop
            if open_trade is not None and open_trade["days"] >= max_hold_days:
                close_trade(close_price)

        # --- Open new trade for tomorrow if none open ---
        opened_today = False
        if open_trade is None and sig.action in ["LONG", "SHORT"] and sig.confidence >= confidence_threshold:
            entry = float(tomorrow_row["Close"])

            # Fee on entry
            capital *= (1.0 - fee)

            open_trade = {
                "direction": sig.action,  # "LONG"/"SHORT"
                "entry": entry,
                "stop": float(sig.levels.get("stop_atr_1x", entry)),
                "target": float(sig.levels.get("tp_atr_2x", entry)),
                "risk": risk_per_trade,
                "days": 0,
                "signal_date": today_hist.index[-1],
                "confidence": float(sig.confidence),
                "score": float(sig.score),
            }
            opened_today = True

        # Log row (date = tomorrow row date, because we apply action into next bar)
        equity_curve.append({
            "Date": tomorrow_row.name,
            "Capital": capital,
            "SignalToday": sig.action,
            "Score": float(sig.score),
            "Confidence": float(sig.confidence),
            "OpenedTrade": int(opened_today),
            "ClosedTrade": int(closed_today),
            "RealizedReturnToday": realized_today,  # already scaled by risk_per_trade
            "HasOpenTrade": int(open_trade is not None),
            "OpenDirection": open_trade["direction"] if open_trade else "",
            "OpenEntry": open_trade["entry"] if open_trade else np.nan,
            "OpenStop": open_trade["stop"] if open_trade else np.nan,
            "OpenTarget": open_trade["target"] if open_trade else np.nan,
        })

    res = pd.DataFrame(equity_curve).set_index("Date")

    # --- Daily summary (latest row) ---
    last = res.iloc[-1]
    prev_cap = float(res["Capital"].iloc[-2]) if len(res) >= 2 else float(last["Capital"])
    today_cap = float(last["Capital"])
    day_ret = today_cap / prev_cap - 1.0
    total_ret = today_cap - 1.0
    mdd = max_drawdown(res["Capital"])
    win_rate = (win_count / trade_count) if trade_count > 0 else 0.0

    print("\n================= DAILY SUMMARY =================")
    print(f"Date:            {res.index[-1].date()}")
    print(f"Signal (today):  {last['SignalToday']} | score={last['Score']:.3f} | conf={last['Confidence']:.2f}")
    print(f"Capital:         {fmt_money(today_cap)}  ({fmt_pct(day_ret)} today, {fmt_pct(total_ret)} total)")
    print(f"Max drawdown:    {fmt_pct(mdd)}")
    print(f"Trades closed:   {trade_count} | Win rate: {win_rate*100:.1f}%")
    if int(last["HasOpenTrade"]) == 1:
        print("Open trade:      YES")
        print(f"  Direction:     {last['OpenDirection']}")
        print(f"  Entry:         {last['OpenEntry']:.2f}")
        print(f"  Stop:          {last['OpenStop']:.2f}")
        print(f"  Target:        {last['OpenTarget']:.2f}")
    else:
        print("Open trade:      NO")
    print("=================================================\n")

    if send_email:
        swings = zigzag_swings(df["Close"], df["ATR14"])
        try:
            mtf_trends = get_mtf_trends()
        except Exception:
            mtf_trends = None
        sig = compute_signal_for_last_day(
            df,
            swings,
            confidence_calibrator=confidence_calibrator,
            mtf_trends=mtf_trends,
        )
        last7 = res.tail(7)
        last7_lines = [
            f"{idx.date()}: {row['SignalToday']} (score={row['Score']:.3f}, conf={row['Confidence']:.2f})"
            for idx, row in last7.iterrows()
        ]
        extra_sections = {
            "Performance": [
                f"Total return since start: {fmt_pct(total_ret)}",
                f"Max drawdown: {fmt_pct(mdd)}",
                f"Trades: {trade_count} | Win rate: {win_rate*100:.1f}%",
            ],
            "Last 7 days actions": last7_lines,
        }
        send_email_notification(sig, extra_sections=extra_sections)

    return res


if __name__ == "__main__":
    equity = forward_simulate(
        start_date="2025-01-01",
        confidence_threshold=0.6,
        send_email=True,
    )

    equity.to_csv("forward_simulation.csv")
    print("Saved: forward_simulation.csv")
