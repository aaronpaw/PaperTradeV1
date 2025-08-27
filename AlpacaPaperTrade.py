# cfv3_runner.py — CFv3 with Alpaca Paper orders + Alpaca data only (IEX or SIP).
# Usage:
#   python cfv3_runner.py --dry-run --scan
#   python cfv3_runner.py --symbols AAPL,NVDA,SPY
#
# Env (required):
#   ALPACA_KEY_ID, ALPACA_SECRET_KEY
# Optional:
#   ALPACA_BASE_URL (defaults to https://paper-api.alpaca.markets)
#
# Deps:
#   pip install alpaca-trade-api pandas pandas_ta pytz

import os, json, math, argparse
from typing import Optional, Iterable
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_ta as ta
import pytz
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="CFv3 hourly runner (Alpaca-only)")
    ap.add_argument("--symbols", type=str, default="AAPL,NVDA,TSLA,AMD,MSFT,META,SPY,QQQ",
                    help="Comma-separated tickers")
    ap.add_argument("--dry-run", action="store_true", help="Do everything except place orders")
    ap.add_argument("--scan", action="store_true", help="Print last 50 entry signals per symbol (no orders)")
    return ap.parse_args()

# ---------------- Env / Globals ----------------
NY = pytz.timezone("America/New_York")

ALPACA_KEY    = "PKLLGZ7JLE6NREWI869C"
ALPACA_SECRET = "Kl6kngWEM19pHFP59JGevAjChZl2rbnadxQ7AWk9"
BASE_URL      = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
FEED          = os.getenv("ALPACA_FEED", "iex")   # "iex" (free) or "sip" (paid)

STATE_PATH    = "state.json"
NEED_BARS_60  = 600      # ask for plenty of history
NEED_BARS_4H  = 240
REQ_COLS: list[str] = ["open","high","low","close","volume"]

def _need_min():
    """Minimal bars needed for indicators to survive dropna."""
    return max(150, SLOW_EMA, BB_LEN, ATR_LEN, RSI_LEN, 120) + 10

# ---------------- Parameters (your row) ----------------
BB_LEN       = 21
BB_STD       = 2.1
BB_TOUCH_LB  = 3
BB_RECLAIM   = "Lower"          # "Lower" or "Middle"

RSI_LEN      = 9
RSI_OS       = 55
REQ_TREND    = True
REQ_SLOPE_UP = False
SLOW_EMA     = 150

ATR_LEN      = 21
ATR_MULT     = 1.8
RR           = 1.8

BLEND_AND_LOOKBACK = 1
COOLDOWN_BARS      = 3
MIN_RVOL           = 1.1
VOL_NORM_MIN       = 0.9
EXCLUDE_FRI        = True
HOLD_CONFIRM       = True
CONFIRM_PRIOR_HIGH = False

USE_AND_SUPPORT    = True
UNION_FALLBACK     = False
PREFERRED_SINGLE   = "BB"       # BB preferred when fallback is needed

MTF_ALIGN          = True       # 60m gated by 4h regime

# Risk / Sizing
RISK_PCT    = 1.0               # % equity risk per trade
MAX_POS_PCT = 25.0              # max position % of equity

# ---------------- State Helpers ----------------
def load_state() -> dict:
    return json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else {}

def save_state(s: dict):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATE_PATH)

# ---------------- Column Normalization ----------------
def ensure_ohlcv(df: pd.DataFrame, symbol: str, context: str = "") -> pd.DataFrame:
    """Return a DataFrame with exactly open/high/low/close/volume in lowercase, or empty if impossible."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Drop symbol level on index if present
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.droplevel(0)
        except Exception:
            pass

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        def _flatten(cols):
            flat = []
            for tup in cols:
                parts = [str(x).strip().lower() for x in tup if x is not None and str(x) != ""]
                flat.append("_".join(parts) if parts else "")
            return flat
        df.columns = _flatten(df.columns)

    df = df.rename(columns=str.lower)

    # If Alpaca sneaks in suffixed names, map them
    sym_l = symbol.lower(); sym_u = symbol.upper()
    out = pd.DataFrame(index=df.index)
    for base in REQ_COLS:
        candidates = [base, f"{base}_{sym_l}", f"{base}_{sym_u}"]
        pick = next((c for c in candidates if c in df.columns), None)
        if pick is None:
            # Last-resort: any base_* where suffix looks like symbol
            for c in df.columns:
                if c.startswith(base + "_") and (c.endswith("_" + sym_l) or c.endswith("_" + sym_u) or c.endswith(sym_l) or c.endswith(sym_u)):
                    pick = c; break
        if pick is None:
            print(f"[GUARD] Missing '{base}' in {context}. Columns seen: {list(df.columns)[:12]}")
            return pd.DataFrame()
        out[base] = df[pick]

    # Normalize tz & order
    if getattr(out.index, "tzinfo", None) is not None:
        out.index = out.index.tz_convert(NY).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out

# ---------------- Alpaca Data (Alpaca-only) ----------------
def fetch_alpaca_bars(api: REST, symbol: str, tf: TimeFrame, limit: int) -> pd.DataFrame:
    """Generic fetch; normalizes to OHLCV."""
    try:
        raw = api.get_bars(symbol, tf, limit=limit, adjustment="raw", feed=FEED).df
    except Exception:
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.index, pd.MultiIndex):
        try:
            raw = raw.xs(symbol, level=0)
        except Exception:
            pass
    return ensure_ohlcv(raw, symbol, context=f"Alpaca {symbol} {tf}")

def fetch_recent_60m(api: REST, symbol: str, need_bars: int = NEED_BARS_60) -> pd.DataFrame:
    need_min = _need_min()

    # 1) Try large limit
    df = fetch_alpaca_bars(api, symbol, TimeFrame(1, TimeFrameUnit.Hour), limit=max(need_bars, 2000))
    if not df.empty:
        print(f"[{symbol}] Alpaca 60m: {len(df)} bars (limit)")
        if len(df) >= need_min:
            return df.tail(max(need_bars, len(df)))

    # 2) Try with explicit start (last 180 days)
    try:
        start = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        raw = api.get_bars(symbol, TimeFrame.Hour, start=start, adjustment="raw", feed=FEED).df
        if raw is not None and not raw.empty:
            if isinstance(raw.index, pd.MultiIndex):
                try:
                    raw = raw.xs(symbol, level=0)
                except Exception:
                    pass
            df2 = ensure_ohlcv(raw, symbol, context=f"Alpaca {symbol} 60m start-180d")
            if not df2.empty:
                print(f"[{symbol}] Alpaca 60m: {len(df2)} bars (start=180d)")
                if len(df2) >= need_min:
                    return df2.tail(max(need_bars, len(df2)))
    except Exception:
        pass

    if not df.empty:
        print(f"[{symbol}] only {len(df)} 60m bars; need >= {need_min}")
    return pd.DataFrame()

def fetch_recent_4h(api: REST, symbol: str, need_bars: int = NEED_BARS_4H) -> pd.DataFrame:
    # Try native 4h first
    df = fetch_alpaca_bars(api, symbol, TimeFrame(4, TimeFrameUnit.Hour), limit=max(need_bars, 1000))
    if not df.empty:
        print(f"[{symbol}] Alpaca 4h: {len(df)} bars")
        return df

    # Fallback: build 4h from 60m
    base = fetch_recent_60m(api, symbol, need_bars*4 + 16)
    if base.empty or not all(c in base.columns for c in REQ_COLS):
        print(f"[{symbol}] 4h fallback skipped; no adequate 60m base (len={len(base) if hasattr(base,'__len__') else 'NA'})")
        return pd.DataFrame()

    res = base.resample("240min", origin="start", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    res = ensure_ohlcv(res, symbol, context=f"Resample4h {symbol}")
    if not res.empty:
        print(f"[{symbol}] Resampled 4h: {len(res)} bars")
    return res

# ---------------- Indicators & Signals ----------------
def build_features_60(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not all(c in df.columns for c in REQ_COLS):
        return pd.DataFrame()

    out = df.copy()

    # BB components
    out[f"sma_{BB_LEN}"] = out["close"].rolling(BB_LEN, min_periods=BB_LEN).mean()
    out[f"std_{BB_LEN}"] = out["close"].rolling(BB_LEN, min_periods=BB_LEN).std(ddof=0)

    # RSI
    out[f"rsi_{RSI_LEN}"] = ta.rsi(out["close"], length=RSI_LEN)

    # Slow EMA (trend)
    ema_slow = out["close"].ewm(span=SLOW_EMA, adjust=False, min_periods=SLOW_EMA).mean()
    out[f"ema_{SLOW_EMA}"] = ema_slow

    # ATR (Series)
    atr_series = ta.atr(out["high"], out["low"], out["close"], length=ATR_LEN)
    if isinstance(atr_series, pd.DataFrame):
        atr_series = atr_series.iloc[:, 0]
    out[f"atr_{ATR_LEN}"] = atr_series

    # RVOL
    vol_sma = out["volume"].rolling(20, min_periods=20).mean()
    out["rvol_20"] = out["volume"] / vol_sma

    # Normalized ATR regime: (ATR/Close)/median120
    atr_rel = (pd.to_numeric(atr_series, errors="coerce") /
               pd.to_numeric(out["close"],  errors="coerce")).astype(float)

    med = atr_rel.rolling(120, min_periods=60).median()
    atr_rel_norm = (atr_rel / med).astype(float)
    atr_rel_norm = atr_rel_norm.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    atr_rel_norm = atr_rel_norm.ffill().fillna(1.0)

    out["atr_rel_norm_120"] = atr_rel_norm

    out = out.dropna()
    return out

def rolling_any(s_bool: pd.Series, lb: int) -> pd.Series:
    return s_bool.rolling(max(lb, 1), min_periods=1).max().astype(bool)

def compute_entries_60(df60: pd.DataFrame, df4h: Optional[pd.DataFrame]) -> pd.Series:
    sma = df60[f"sma_{BB_LEN}"]; std = df60[f"std_{BB_LEN}"]
    bb_b = sma
    bb_l = sma - BB_STD * std
    touch_win = rolling_any(df60["low"] < bb_l, BB_TOUCH_LB)
    reclaim_level = bb_l if BB_RECLAIM == "Lower" else bb_b
    e_bb_base = touch_win & (df60["close"] > reclaim_level)

    rsi = df60[f"rsi_{RSI_LEN}"]
    rsi_cross_up = (rsi.shift(1) < RSI_OS) & (rsi >= RSI_OS)

    if REQ_TREND:
        ema = df60[f"ema_{SLOW_EMA}"]
        slope_ok = True if not REQ_SLOPE_UP else (ema.diff() > 0)
        trend_ok = (df60["close"] > ema) & slope_ok
    else:
        trend_ok = pd.Series(True, index=df60.index)

    e_rsi_base = rsi_cross_up & trend_ok

    # Hold confirm (use pandas nullable boolean to avoid warnings)
    if HOLD_CONFIRM:
        e_bb_sig  = e_bb_base.shift(1).astype("boolean").fillna(False) & (df60["close"] > reclaim_level)
        e_rsi_sig = e_rsi_base.shift(1).astype("boolean").fillna(False) & (df60["close"] > df60["close"].shift(1))
    else:
        e_bb_sig, e_rsi_sig = e_bb_base, e_rsi_base

    if CONFIRM_PRIOR_HIGH:
        ph = df60["close"] > df60["high"].shift(1)
        e_bb_sig  = e_bb_sig & ph
        e_rsi_sig = e_rsi_sig & ph

    # Gates
    regime_ok = (df60["atr_rel_norm_120"] >= VOL_NORM_MIN)
    rvol_ok   = (df60["rvol_20"] >= MIN_RVOL)
    if EXCLUDE_FRI:
        not_fri = df60.index.dayofweek != 4
        e_bb_sig  = e_bb_sig & not_fri
        e_rsi_sig = e_rsi_sig & not_fri
    e_bb_sig  = (e_bb_sig  & regime_ok & rvol_ok).astype(bool)
    e_rsi_sig = (e_rsi_sig & regime_ok & rvol_ok).astype(bool)

    # MTF (4h) regime mapped to 60m
    if MTF_ALIGN and df4h is not None and not df4h.empty and "ema_slow" in df4h.columns:
        ema4 = df4h["ema_slow"]
        regime4 = (df4h["close"] > ema4)
        if REQ_SLOPE_UP:
            regime4 = regime4 & (ema4.diff() > 0)
        regime4_on_60 = regime4.reindex(df60.index, method="ffill").astype("boolean").fillna(False)
        e_bb_sig  = e_bb_sig & regime4_on_60
        e_rsi_sig = e_rsi_sig & regime4_on_60

    # Combine
    sup_rsi = rolling_any(e_rsi_sig, BLEND_AND_LOOKBACK)
    sup_bb  = rolling_any(e_bb_sig,  BLEND_AND_LOOKBACK)
    entries_and = (e_bb_sig & sup_rsi) | (e_rsi_sig & sup_bb)

    if USE_AND_SUPPORT:
        if UNION_FALLBACK:
            entries = (entries_and | e_bb_sig | e_rsi_sig).astype(bool)
        else:
            preferred = e_rsi_sig if PREFERRED_SINGLE.upper() == "RSI" else e_bb_sig
            entries = (entries_and | (preferred & (~entries_and))).astype(bool)
    else:
        entries = (e_bb_sig | e_rsi_sig).astype(bool)

    warm = _need_min()
    if len(entries) >= warm:
        entries.iloc[:warm] = False
    return entries.astype(bool)

# ---------------- Broker Helpers ----------------
def is_flat(api: REST, symbol: str) -> bool:
    try:
        _ = api.get_position(symbol)
        return False
    except Exception:
        pass
    try:
        open_orders = [o for o in api.list_orders(status="open") if o.symbol == symbol]
        return len(open_orders) == 0
    except Exception:
        return True

# ---------------- Debug / Scan ----------------
def debug_last_signals(symbol: str, df60: pd.DataFrame, df4h: Optional[pd.DataFrame], n=50):
    entries = compute_entries_60(df60, df4h)
    hits = entries.iloc[:-1].tail(n)
    print(f"[{symbol}] last {n} closed bars with entries:")
    any_hit = False
    for ts, v in hits[hits].items():
        print("  •", ts)
        any_hit = True
    if not any_hit:
        print("  (none)")

# ---------------- Trading Routine ----------------
def on_bar_closed(symbol: str, api: REST, state: dict, df60: pd.DataFrame, df4h: Optional[pd.DataFrame], dry_run: bool):
    if df60.empty or not all(c in df60.columns for c in REQ_COLS):
        print(f"[{symbol}] invalid 60m data; skip")
        return

    entries = compute_entries_60(df60, df4h)
    if entries.empty or len(entries) < 2:
        print(f"[{symbol}] entries not ready")
        return

    bar_t   = df60.index[-2]                # last CLOSED bar
    px      = float(df60["close"].iloc[-2])
    atr     = float(df60[f"atr_{ATR_LEN}"].iloc[-2])

    # cooldown per symbol
    s = state.get(symbol, {})
    last_iso = s.get("last_entry_iso")
    cooldown_ok = True
    if last_iso:
        try:
            bars_since = df60.index.get_loc(bar_t) - df60.index.get_loc(pd.to_datetime(last_iso))
            cooldown_ok = bars_since >= COOLDOWN_BARS
        except Exception:
            cooldown_ok = True

    flat = is_flat(api, symbol)

    print(f"[{symbol}] bar={bar_t}, close={px:.2f}, atr={atr:.4f}, flat={flat}, cooldown_ok={cooldown_ok}, entry={bool(entries.iloc[-2])}")

    if entries.iloc[-2] and cooldown_ok and flat:
        acct = api.get_account()
        equity = float(acct.equity)
        risk_cash = equity * (RISK_PCT / 100.0)
        sl_pct = max(ATR_MULT * (atr / px), 1e-6)
        tp_pct = max(RR * sl_pct, 1e-6)

        base_qty = risk_cash / (sl_pct * px)
        max_shares = (equity * (MAX_POS_PCT/100.0)) / px
        qty = max(0, math.floor(min(base_qty, max_shares)))

        if qty <= 0:
            print(f"[{symbol}] qty=0 (equity={equity:.2f}, px={px:.2f}); skip")
            return

        stop_price = round(px * (1 - sl_pct), 2)
        tp_price   = round(px * (1 + tp_pct), 2)

        msg = f"[{symbol}] BUY {qty} @~{px:.2f} SL {stop_price} TP {tp_price} (bar {bar_t})"
        if dry_run:
            print(msg, "(DRY RUN)")
        else:
            print(msg)
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                type="market",
                time_in_force="gtc",
                order_class="bracket",
                take_profit={"limit_price": tp_price},
                stop_loss={"stop_price": stop_price}
            )
            s["last_entry_iso"] = bar_t.isoformat()
            state[symbol] = s
            save_state(state)
    else:
        why = []
        if not entries.iloc[-2]: why.append("no signal")
        if not cooldown_ok: why.append("cooldown")
        if not flat: why.append("not flat")
        print(f"[{symbol}] no trade ({', '.join(why)})")

# ---------------- Main ----------------
def main():
    # Optional: opt-in to pandas “future” behavior to silence warnings globally
    pd.set_option("future.no_silent_downcasting", True)

    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not (ALPACA_KEY and ALPACA_SECRET):
        raise SystemExit("Set ALPACA_KEY_ID and ALPACA_SECRET_KEY env vars.")

    api = REST(ALPACA_KEY, ALPACA_SECRET, base_url=BASE_URL)
    state = load_state()

    # Account summary
    try:
        acct = api.get_account()
        print(f"Account status: {acct.status} | buying_power: {acct.buying_power} | blocked: {acct.account_blocked}")
    except Exception as e:
        print(f"[WARN] Could not read account: {e}")

    for sym in symbols:
        try:
            df60 = fetch_recent_60m(api, sym, NEED_BARS_60)
            if df60.empty:
                print(f"[{sym}] no 60m data from Alpaca; skip")
                continue
            print(f"[{sym}] fetched {len(df60)} x 60m bars pre-features")

            df60 = build_features_60(df60)
            if df60.empty:
                print(f"[{sym}] features empty after build (insufficient warmup); skip")
                continue
            print(f"[{sym}] features rows after warmup: {len(df60)}")

            df4h = None
            if MTF_ALIGN:
                raw4 = fetch_recent_4h(api, sym, NEED_BARS_4H)
                if not raw4.empty:
                    df4h = ensure_ohlcv(raw4, sym, context=f"{sym} 4h")
                    if not df4h.empty:
                        df4h["ema_slow"] = df4h["close"].ewm(span=SLOW_EMA, adjust=False, min_periods=SLOW_EMA).mean()
                        df4h = df4h.dropna()
                        print(f"[{sym}] 4h rows after ema: {len(df4h)}")
                # else: df4h stays None

            if args.scan:
                debug_last_signals(sym, df60, df4h, n=50)

            on_bar_closed(sym, api, state, df60, df4h, dry_run=args.dry_run)

        except Exception as e:
            print(f"[{sym}] ERROR: {e}")

if __name__ == "__main__":
    main()
