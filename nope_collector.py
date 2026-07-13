#!/usr/bin/env python3
"""
BLW NOPE v1 — daily collector (free data via yfinance)
======================================================
Computes NOPE (Net Options Pricing Effect, Lily Francus) for SPY and appends
one row per trading day to three Pine Seeds CSV files:

    data/NOPE_ALL.csv       all expirations (DTE <= MAX_DTE)
    data/NOPE_EX0DTE.csv    excluding same-day expiry
    data/NOPE_0DTE.csv      same-day expiry only

Conventions (FROZEN — do not change without re-baselining the MAD history):
  * Contract multiplier x100 IS included.
        NOPE = 100 * (sum(callVol*callDelta) - sum(putVol*|putDelta|)) / SPY share volume
    i.e. NOPE is a share-equivalent ratio (can exceed 1.0 in the 0DTE era).
  * Delta = Black-Scholes delta computed from Yahoo's per-contract IV,
    with dividend yield q and risk-free r (from ^IRX, fallback constant).
  * Snapshot is taken near the close (default gate 15:40-16:05 ET) so that
    0DTE contracts are still alive. Volume is near-full-day.
  * CSV format is Pine Seeds EOD: YYYYMMDDT,open,high,low,close,volume
    with open=high=low=close=NOPE and volume = SPY share volume.

Usage:
    python nope_collector.py            # only runs inside the ET time gate
    python nope_collector.py --force    # skip the time gate (backtesting a
                                        # snapshot off-hours will use stale
                                        # deltas — fine for pipeline tests,
                                        # not for production rows)
"""

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

# ---------------------------------------------------------------- settings
UNDERLYING      = "SPY"
MAX_DTE         = 60          # ignore expirations beyond this (thin, slow to fetch)
FALLBACK_RATE   = 0.045       # risk-free if ^IRX fetch fails
DIV_YIELD       = 0.012       # SPY approximate dividend yield
IV_MIN, IV_MAX  = 0.005, 5.0  # sanity band for Yahoo IV
T_FLOOR_YEARS   = 10.0 / (60 * 24 * 365)   # 10 minutes, avoids div-by-zero on 0DTE
GATE_START      = (15, 40)    # ET window in which production runs are allowed
GATE_END        = (16, 5)
DATA_DIR        = Path(__file__).resolve().parent / "data"
ET              = ZoneInfo("America/New_York")

BUCKETS = ("NOPE_ALL", "NOPE_EX0DTE", "NOPE_0DTE")


# ---------------------------------------------------------------- helpers
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(spot, strike, t_years, iv, r, q, is_call):
    """Black-Scholes delta with continuous dividend yield."""
    t = max(t_years, T_FLOOR_YEARS)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    if is_call:
        return math.exp(-q * t) * norm_cdf(d1)
    return math.exp(-q * t) * (norm_cdf(d1) - 1.0)


def get_risk_free() -> float:
    try:
        irx = yf.Ticker("^IRX").fast_info["last_price"]  # 13-week T-bill, percent
        if irx and 0.0 < irx < 15.0:
            return irx / 100.0
    except Exception:
        pass
    return FALLBACK_RATE


def in_time_gate(now_et: datetime) -> bool:
    hm = (now_et.hour, now_et.minute)
    return now_et.weekday() < 5 and GATE_START <= hm <= GATE_END


def clean_iv(df, label, stats):
    """Replace out-of-band / missing IV with the median IV of the same
    expiry+type; drop rows only if no usable IV exists at all."""
    iv = df["impliedVolatility"]
    bad = iv.isna() | (iv < IV_MIN) | (iv > IV_MAX)
    stats[f"{label}_bad_iv"] = int(bad.sum())
    good_median = iv[~bad].median()
    if math.isnan(good_median) if isinstance(good_median, float) else good_median is None:
        return df[~bad]  # nothing to impute from; drop bad rows
    df = df.copy()
    df.loc[bad, "impliedVolatility"] = good_median
    return df


# ---------------------------------------------------------------- core
def collect(force: bool = False) -> int:
    now_et = datetime.now(timezone.utc).astimezone(ET)
    if not force and not in_time_gate(now_et):
        print(f"[skip] {now_et:%Y-%m-%d %H:%M %Z} outside snapshot gate "
              f"{GATE_START[0]:02d}:{GATE_START[1]:02d}-{GATE_END[0]:02d}:{GATE_END[1]:02d} ET")
        return 0

    tkr = yf.Ticker(UNDERLYING)
    spot = tkr.fast_info["last_price"]
    day_bar = tkr.history(period="1d")
    share_volume = float(day_bar["Volume"].iloc[-1])
    if not spot or share_volume <= 0:
        print("[error] could not fetch spot / share volume", file=sys.stderr)
        return 1

    r = get_risk_free()
    today = now_et.date()
    close_hour = 16

    acc = {b: {"call": 0.0, "put": 0.0, "contracts": 0} for b in BUCKETS}
    stats = {}

    expiries = [e for e in tkr.options
                if (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= MAX_DTE]
    if not expiries:
        print("[error] no expirations returned", file=sys.stderr)
        return 1

    for exp in expiries:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        exp_dt = datetime(exp_date.year, exp_date.month, exp_date.day,
                          close_hour, 0, tzinfo=ET)
        t_years = (exp_dt - now_et).total_seconds() / (365.0 * 24 * 3600)
        if t_years <= 0 and dte > 0:
            continue  # defensive; should not happen

        try:
            chain = tkr.option_chain(exp)
        except Exception as e:
            print(f"[warn] chain fetch failed for {exp}: {e}", file=sys.stderr)
            continue

        for df, is_call in ((chain.calls, True), (chain.puts, False)):
            df = df[["strike", "volume", "impliedVolatility"]].copy()
            df["volume"] = df["volume"].fillna(0)
            df = df[df["volume"] > 0]
            if df.empty:
                continue
            df = clean_iv(df, f"{exp}_{'C' if is_call else 'P'}", stats)

            for row in df.itertuples(index=False):
                delta = bs_delta(spot, row.strike, t_years,
                                 row.impliedVolatility, r, DIV_YIELD, is_call)
                dv = row.volume * abs(delta)
                side = "call" if is_call else "put"
                targets = ["NOPE_ALL", "NOPE_0DTE" if dte == 0 else "NOPE_EX0DTE"]
                for b in targets:
                    acc[b][side] += dv
                    acc[b]["contracts"] += int(row.volume)

    # ------------------------------------------------------------ write
    DATA_DIR.mkdir(exist_ok=True)
    date_key = f"{today:%Y%m%d}T"
    for b in BUCKETS:
        nope = 100.0 * (acc[b]["call"] - acc[b]["put"]) / share_volume
        line = f"{date_key},{nope:.6f},{nope:.6f},{nope:.6f},{nope:.6f},{int(share_volume)}"
        path = DATA_DIR / f"{b}.csv"
        rows = []
        if path.exists():
            rows = [l for l in path.read_text().splitlines()
                    if l.strip() and not l.startswith(date_key)]  # replace same-day row
        rows.append(line)
        path.write_text("\n".join(rows) + "\n")
        print(f"[ok] {b:<12} NOPE={nope:+.4f}  "
              f"callDV={acc[b]['call']:,.0f}  putDV={acc[b]['put']:,.0f}  "
              f"contracts={acc[b]['contracts']:,}")

    bad_total = sum(v for k, v in stats.items())
    print(f"[quality] spot={spot:.2f}  shareVol={share_volume:,.0f}  r={r:.4f}  "
          f"expiries={len(expiries)}  IV imputed on {bad_total} contract rows  "
          f"snapshot={now_et:%Y-%m-%d %H:%M %Z}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="bypass the 15:40-16:05 ET time gate")
    sys.exit(collect(force=ap.parse_args().force))
