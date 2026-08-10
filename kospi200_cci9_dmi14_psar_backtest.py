#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KOSPI200 fixed-current-universe backtest
----------------------------------------
Period default: 2023-08-10 ~ 2026-08-10
Strategy:
  Entry signal at today's close:
    1) CCI(9) crosses above 0
    2) Parabolic SAR bullish (Close > PSAR)
    3) +DI(14) > -DI(14)
  Fill: next trading day's OPEN

  Exit signal at today's close if ANY:
    1) CCI(9) < 0
    2) Parabolic SAR bearish (Close < PSAR)
    3) +DI(14) <= -DI(14)
  Fill: next trading day's OPEN

PSAR:
  Standard acceleration factor step=0.02, max=0.20.
  "14-day SAR" is not a standard PSAR parameter; --warmup 14 is used.

Universe:
  Survivorship bias intentionally ignored.
  Current KOSPI200 universe is frozen across the whole backtest period.

Input modes:
  A) Local FinanceData/marcap yearly parquet files:
       --marcap-dir ./data
       files: marcap-2023.parquet ... marcap-2026.parquet
     Universe can be:
       --universe-csv kospi200_codes.csv
     where CSV has a Code column (6-digit ticker).
     OR, if internet is available, pykrx can fetch current KOSPI200.

  B) Online FinanceDataReader:
       --online
     Requires finance-datareader and internet.

Outputs:
  output/summary.csv
  output/by_stock.csv
  output/trades.csv
  output/portfolio_daily.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def cci(df: pd.DataFrame, period: int = 9) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()
    md = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    denom = 0.015 * md
    out = (tp - ma) / denom.replace(0, np.nan)
    return out


def dmi_wilder(df: pd.DataFrame, period: int = 14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()

    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)

    tr = pd.concat(
        [
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder RMA = EMA(alpha=1/period, adjust=False)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_sm = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_sm / atr.replace(0, np.nan)
    minus_di = 100 * minus_sm / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return plus_di, minus_di, adx


def psar(df: pd.DataFrame, step: float = 0.02, max_af: float = 0.20) -> pd.Series:
    """Classic Parabolic SAR."""
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float, index=df.index)
    if n == 1:
        return pd.Series([low[0]], index=df.index)

    sar = np.full(n, np.nan)
    bull = close[1] >= close[0]
    af = step
    ep = high[0] if bull else low[0]
    sar[0] = low[0] if bull else high[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        cur = prev_sar + af * (ep - prev_sar)

        if bull:
            if i >= 2:
                cur = min(cur, low[i - 1], low[i - 2])
            else:
                cur = min(cur, low[i - 1])

            if low[i] < cur:  # reversal to bear
                bull = False
                cur = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(max_af, af + step)
        else:
            if i >= 2:
                cur = max(cur, high[i - 1], high[i - 2])
            else:
                cur = max(cur, high[i - 1])

            if high[i] > cur:  # reversal to bull
                bull = True
                cur = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(max_af, af + step)

        sar[i] = cur

    return pd.Series(sar, index=df.index, name="PSAR")


def calc_indicators(df: pd.DataFrame, cci_n=9, dmi_n=14, psar_step=0.02, psar_max=0.20):
    x = df.copy()
    x["CCI"] = cci(x, cci_n)
    x["+DI"], x["-DI"], x["ADX"] = dmi_wilder(x, dmi_n)
    x["PSAR"] = psar(x, psar_step, psar_max)
    x["PSAR_BULL"] = x["Close"] > x["PSAR"]
    x["CCI_CROSS_UP_0"] = (x["CCI"] > 0) & (x["CCI"].shift(1) <= 0)
    x["ENTRY_SIGNAL"] = x["CCI_CROSS_UP_0"] & x["PSAR_BULL"] & (x["+DI"] > x["-DI"])
    x["EXIT_SIGNAL"] = (x["CCI"] < 0) | (~x["PSAR_BULL"]) | (x["+DI"] <= x["-DI"])
    return x


def backtest_one(
    df: pd.DataFrame,
    code: str,
    name: str = "",
    start="2023-08-10",
    end="2026-08-10",
    fee_side=0.0,
    warmup=14,
):
    x = df.sort_index().copy()
    x = x[~x.index.duplicated(keep="last")]
    required = ["Open", "High", "Low", "Close"]
    x = x.dropna(subset=required)
    x = x[(x[required] > 0).all(axis=1)]

    # Calculate indicators on all available history, then clip evaluation window.
    x = calc_indicators(x)
    eval_mask = (x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))
    idx_eval = np.flatnonzero(eval_mask.to_numpy())
    if len(idx_eval) < 2:
        return None, pd.DataFrame(), pd.DataFrame()

    first_eval = idx_eval[0]
    # At least warmup bars before first allowed signal.
    first_signal_i = max(first_eval, warmup)

    equity = 1.0
    position = False
    entry_price = None
    entry_date = None
    pending = None  # "BUY" / "SELL"
    trades = []
    daily_rows = []

    prev_equity = equity
    shares = 0.0
    cash = 1.0

    for i in range(first_eval, len(x)):
        dt = x.index[i]
        if dt > pd.Timestamp(end):
            break
        row = x.iloc[i]

        # Execute prior close signal at today's open.
        if pending == "BUY" and not position:
            px = float(row["Open"]) * (1 + fee_side)
            if px > 0:
                shares = cash / px
                cash = 0.0
                position = True
                entry_price = px
                entry_date = dt
        elif pending == "SELL" and position:
            px = float(row["Open"]) * (1 - fee_side)
            cash = shares * px
            gross_ret = (float(row["Open"]) / (entry_price / (1 + fee_side))) - 1 if entry_price else np.nan
            net_ret = px / entry_price - 1 if entry_price else np.nan
            trades.append({
                "Code": code, "Name": name,
                "EntryDate": entry_date, "ExitDate": dt,
                "EntryPriceNet": entry_price, "ExitPriceNet": px,
                "Return": net_ret, "GrossApprox": gross_ret,
                "HoldingDays": (dt - entry_date).days if entry_date is not None else np.nan
            })
            shares = 0.0
            position = False
            entry_price = None
            entry_date = None
        pending = None

        # Mark to close.
        equity = cash if not position else shares * float(row["Close"])
        daily_ret = equity / prev_equity - 1 if prev_equity > 0 else 0.0
        daily_rows.append({"Date": dt, "Code": code, "Equity": equity, "Return": daily_ret, "Position": int(position)})
        prev_equity = equity

        # Generate close signal for next open.
        if i >= first_signal_i and i + 1 < len(x) and x.index[i + 1] <= pd.Timestamp(end):
            if not position and bool(row["ENTRY_SIGNAL"]):
                pending = "BUY"
            elif position and bool(row["EXIT_SIGNAL"]):
                pending = "SELL"

    # Liquidate open position at final close, charging sell-side fee.
    if position:
        last_dt = daily_rows[-1]["Date"]
        last_close = float(x.loc[last_dt, "Close"])
        px = last_close * (1 - fee_side)
        cash = shares * px
        net_ret = px / entry_price - 1 if entry_price else np.nan
        trades.append({
            "Code": code, "Name": name,
            "EntryDate": entry_date, "ExitDate": last_dt,
            "EntryPriceNet": entry_price, "ExitPriceNet": px,
            "Return": net_ret, "GrossApprox": np.nan,
            "HoldingDays": (last_dt - entry_date).days if entry_date is not None else np.nan
        })
        # overwrite final marked equity to liquidation net value
        daily_rows[-1]["Equity"] = cash
        if len(daily_rows) >= 2:
            daily_rows[-1]["Return"] = cash / daily_rows[-2]["Equity"] - 1
        else:
            daily_rows[-1]["Return"] = cash - 1

    daily = pd.DataFrame(daily_rows).set_index("Date")
    tdf = pd.DataFrame(trades)

    if daily.empty:
        return None, tdf, daily

    total_ret = daily["Equity"].iloc[-1] - 1
    days = max((daily.index[-1] - daily.index[0]).days, 1)
    years = days / 365.25
    cagr = (daily["Equity"].iloc[-1] ** (1 / years) - 1) if years > 0 and daily["Equity"].iloc[-1] > 0 else np.nan

    peak = daily["Equity"].cummax()
    dd = daily["Equity"] / peak - 1
    mdd = dd.min()

    r = daily["Return"].replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan

    if len(tdf):
        win_rate = (tdf["Return"] > 0).mean()
        avg_trade = tdf["Return"].mean()
        median_trade = tdf["Return"].median()
    else:
        win_rate = avg_trade = median_trade = np.nan

    # Buy & Hold: first evaluation open -> last close, same fee model.
    eval_x = x.loc[(x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))]
    bh_entry = float(eval_x["Open"].iloc[0]) * (1 + fee_side)
    bh_exit = float(eval_x["Close"].iloc[-1]) * (1 - fee_side)
    bh_ret = bh_exit / bh_entry - 1

    stats = {
        "Code": code,
        "Name": name,
        "Start": daily.index[0],
        "End": daily.index[-1],
        "Bars": len(daily),
        "StrategyReturn": total_ret,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "WinRate": win_rate,
        "AvgTradeReturn": avg_trade,
        "MedianTradeReturn": median_trade,
        "Trades": len(tdf),
        "BuyHoldReturn": bh_ret,
        "ExcessVsBuyHold": total_ret - bh_ret,
    }
    return stats, tdf, daily


def load_universe_csv(path: str):
    u = pd.read_csv(path, dtype={"Code": str})
    if "Code" not in u.columns:
        raise ValueError("Universe CSV must contain a 'Code' column.")
    u["Code"] = u["Code"].astype(str).str.zfill(6)
    if "Name" not in u.columns:
        u["Name"] = ""
    return u[["Code", "Name"]].drop_duplicates("Code").reset_index(drop=True)


def fetch_current_kospi200_online():
    import FinanceDataReader as fdr
    u = fdr.SnapDataReader("KRX/INDEX/STOCK/1028").copy()
    if u.empty or "Code" not in u.columns:
        raise RuntimeError("Could not fetch current KOSPI200 constituents via FinanceDataReader.")
    u["Code"] = u["Code"].astype(str).str.zfill(6)
    if "Name" not in u.columns:
        u["Name"] = ""
    u = u[["Code", "Name"]].drop_duplicates("Code").reset_index(drop=True)
    if len(u) != 200:
        raise RuntimeError(f"Expected 200 KOSPI200 constituents, got {len(u)}")
    return u


def load_marcap(marcap_dir: str, start: str, end: str):
    d = Path(marcap_dir)
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    parts = []
    for y in range(y0, y1 + 1):
        p = d / f"marcap-{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        z = pd.read_parquet(p)
        if "Date" in z.columns:
            z["Date"] = pd.to_datetime(z["Date"])
        else:
            z = z.reset_index()
            z["Date"] = pd.to_datetime(z["Date"])
        z["Code"] = z["Code"].astype(str).str.zfill(6)
        parts.append(z)
    allp = pd.concat(parts, ignore_index=True)
    return allp


def online_price(code: str, start: str, end: str):
    import FinanceDataReader as fdr
    z = fdr.DataReader(code, start, end).copy()
    z.index = pd.to_datetime(z.index)
    cols = {c.lower(): c for c in z.columns}
    # FinanceDataReader normally already uses canonical capitalization.
    need = ["Open", "High", "Low", "Close", "Volume"]
    for c in need:
        if c not in z.columns:
            alt = cols.get(c.lower())
            if alt:
                z[c] = z[alt]
    return z


def aggregate_portfolio(dailies: dict[str, pd.DataFrame], start: str, end: str):
    """
    Equal-weight across available frozen-universe names.
    Each stock starts at equity 1.0. Missing dates are treated as zero return.
    This is equivalent to equal initial allocation with idle cash for names
    that have not listed yet / lack data.
    """
    all_dates = pd.date_range(start, end, freq="D")
    rets = []
    for code, d in dailies.items():
        s = d["Return"].reindex(all_dates).fillna(0.0)
        s.name = code
        rets.append(s)
    if not rets:
        return pd.DataFrame()
    mat = pd.concat(rets, axis=1)
    port_ret = mat.mean(axis=1)
    # remove weekends/holidays where every return is zero only if desired;
    # keep them from affecting Sharpe by filtering on days with at least one original trading observation
    active = pd.Series(False, index=all_dates)
    for d in dailies.values():
        active.loc[active.index.intersection(d.index)] = True
    out = pd.DataFrame({"Return": port_ret, "Active": active})
    out = out[out["Active"]].drop(columns="Active")
    out["Equity"] = (1 + out["Return"]).cumprod()
    return out


def metrics_from_equity(d: pd.DataFrame):
    if d.empty:
        return {}
    total = d["Equity"].iloc[-1] - 1
    days = max((d.index[-1] - d.index[0]).days, 1)
    years = days / 365.25
    cagr = d["Equity"].iloc[-1] ** (1/years) - 1
    dd = d["Equity"] / d["Equity"].cummax() - 1
    r = d["Return"].dropna()
    sharpe = r.mean()/r.std(ddof=1)*np.sqrt(252) if len(r)>1 and r.std(ddof=1)>0 else np.nan
    return {"TotalReturn": total, "CAGR": cagr, "MDD": dd.min(), "Sharpe": sharpe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-08-10")
    ap.add_argument("--end", default="2026-08-10")
    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--marcap-dir", default=None)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--fee-side", type=float, default=0.0,
                    help="Per-side proportional cost. Example: 0.001 = 0.10%% each side.")
    ap.add_argument("--warmup", type=int, default=14)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    if not args.online and not args.marcap_dir:
        raise SystemExit("Choose --online or --marcap-dir.")

    if args.universe_csv:
        universe = load_universe_csv(args.universe_csv)
    else:
        universe = fetch_current_kospi200_online()

    if len(universe) != 200:
        print(f"WARNING: universe count is {len(universe)}, expected 200.")

    marcap = None
    if args.marcap_dir:
        marcap = load_marcap(args.marcap_dir, args.start, args.end)

    all_stats, all_trades, dailies = [], [], {}
    for j, row in universe.iterrows():
        code, name = row["Code"], row["Name"]
        try:
            if marcap is not None:
                z = marcap[marcap["Code"] == code].copy()
                z = z.set_index("Date").sort_index()
            else:
                z = online_price(code, args.start, args.end)

            stats, trades, daily = backtest_one(
                z, code, name, args.start, args.end,
                fee_side=args.fee_side, warmup=args.warmup
            )
            if stats is not None:
                all_stats.append(stats)
                dailies[code] = daily
                if not trades.empty:
                    all_trades.append(trades)
            print(f"[{j+1:3d}/{len(universe)}] {code} {name}: OK")
        except Exception as e:
            print(f"[{j+1:3d}/{len(universe)}] {code} {name}: ERROR {e}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    by_stock = pd.DataFrame(all_stats)
    by_stock.to_csv(outdir/"by_stock.csv", index=False, encoding="utf-8-sig")
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(outdir/"trades.csv", index=False, encoding="utf-8-sig")

    portfolio = aggregate_portfolio(dailies, args.start, args.end)
    portfolio.to_csv(outdir/"portfolio_daily.csv", encoding="utf-8-sig")

    pm = metrics_from_equity(portfolio)
    summary = {
        "UniverseRequested": 200,
        "StocksWithUsableData": len(by_stock),
        "PeriodStart": args.start,
        "PeriodEnd": args.end,
        "FeePerSide": args.fee_side,
        **{f"Portfolio_{k}": v for k, v in pm.items()},
        "CrossSection_MeanStrategyReturn": by_stock["StrategyReturn"].mean() if len(by_stock) else np.nan,
        "CrossSection_MedianStrategyReturn": by_stock["StrategyReturn"].median() if len(by_stock) else np.nan,
        "CrossSection_MeanCAGR": by_stock["CAGR"].mean() if len(by_stock) else np.nan,
        "CrossSection_MeanMDD": by_stock["MDD"].mean() if len(by_stock) else np.nan,
        "CrossSection_MeanSharpe": by_stock["Sharpe"].mean() if len(by_stock) else np.nan,
        "TradeWinRate_AllTrades": (trades["Return"] > 0).mean() if len(trades) else np.nan,
        "AvgTradeReturn_AllTrades": trades["Return"].mean() if len(trades) else np.nan,
        "TotalTrades": len(trades),
        "CrossSection_MeanBuyHoldReturn": by_stock["BuyHoldReturn"].mean() if len(by_stock) else np.nan,
        "StrategyBeatsBuyHoldPct": (by_stock["ExcessVsBuyHold"] > 0).mean() if len(by_stock) else np.nan,
    }
    pd.DataFrame([summary]).to_csv(outdir/"summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
