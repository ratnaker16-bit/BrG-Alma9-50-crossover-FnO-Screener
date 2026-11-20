import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="F&O ALMA Screener", layout="wide")

# ---------------- F&O stock list (hardcoded; update when NSE revises the list) ----------------
FNO_STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS",
    "ALKEM", "AMBUJACEM", "ANGELONE", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "ATGL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BEL", "BERGEPAINT",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSE",
    "BSOFT", "CAMS", "CANBK", "CDSL", "CESC", "CGPOWER", "CHAMBLFERT", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUMMINSIND", "CYIENT", "DABUR",
    "DALBHARAT", "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY", "EICHERMOT",
    "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK", "GMRAIRPORT", "GODREJCP",
    "GODREJPROP", "GRANULES", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HFCL", "HINDALCO", "HINDCOPPER", "HINDPETRO", "HINDUNILVR",
    "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL",
    "INDHOTEL", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY", "INOXWIND", "IOC", "IRB",
    "IRCTC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN", "JSWENERGY", "JSWSTEEL",
    "JUBLFOOD", "KALYANKJIL", "KEI", "KOTAKBANK", "KPITTECH", "LAURUSLABS", "LICHSGFIN",
    "LICI", "LODHA", "LT", "LTF", "LTIM", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MANKIND",
    "MARICO", "MARUTI", "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "MGL", "MOTHERSON",
    "MPHASIS", "MUTHOOTFIN", "NATIONALUM", "NAUKRI", "NBCC", "NCC", "NESTLEIND", "NHPC",
    "NMDC", "NTPC", "NYKAA", "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI",
    "PAYTM", "PEL", "PERSISTENT", "PETRONET", "PFC", "PHOENIXLTD", "PIDILITIND", "PIIND",
    "PNB", "PNBHOUSING", "POLICYBZR", "POLYCAB", "POONAWALLA", "POWERGRID", "PRESTIGE",
    "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SRF", "SUNPHARMA", "SUPREMEIND",
    "SYNGENE", "TATACHEM", "TATACOMM", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TCS", "TECHM", "TIINDIA", "TITAGARH", "TITAN", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TVSMOTOR", "UNIONBANK", "UNITDSPR", "UNOMINDA", "UPL",
    "ULTRACEMCO", "VBL", "VEDL", "VOLTAS", "WIPRO", "YESBANK", "ZYDUSLIFE",
]


def to_ticker(sym: str) -> str:
    return f"{sym}.NS"


# ---------------- ALMA ----------------
def alma(series: pd.Series, window: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average (same formula as TradingView, defaults offset=0.85, sigma=6)."""
    m = offset * (window - 1)
    s = window / sigma
    idx = np.arange(window)
    w = np.exp(-((idx - m) ** 2) / (2 * s * s))
    w /= w.sum()
    vals = series.to_numpy(dtype=float)
    if len(vals) < window:
        return pd.Series(np.nan, index=series.index)
    out = np.full(len(vals), np.nan)
    out[window - 1:] = np.convolve(vals, w[::-1], mode="valid")
    return pd.Series(out, index=series.index)


# ---------------- Data ----------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch(interval: str, symbols: tuple) -> dict:
    tickers = [to_ticker(s) for s in symbols]
    raw = yf.download(
        tickers, period="5d", interval=interval, group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )
    result = {}
    for s, t in zip(symbols, tickers):
        try:
            df = raw[t][["Close"]].dropna()
            if len(df):
                result[s] = df
        except Exception:
            continue
    return result


def check_cross(df: pd.DataFrame, interval_min: int, fast: int, slow: int,
                lookback: int, use_closed_only: bool):
    """Return ('CE'|'PE'|None, last_close, alma_fast, alma_slow, cross_time)."""
    d = df.copy()
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    d.index = d.index.tz_convert("Asia/Kolkata")

    if use_closed_only and len(d) > 1:
        # last row from yfinance is the still-forming candle -> drop it
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        if d.index[-1] + pd.Timedelta(minutes=interval_min) > now:
            d = d.iloc[:-1]

    if len(d) < slow + lookback + 1:
        return None

    close = d["Close"]
    a_fast = alma(close, fast)
    a_slow = alma(close, slow)
    diff = a_fast - a_slow

    n = len(diff)
    for k in range(lookback):
        i = n - 1 - k
        prev, cur = diff.iloc[i - 1], diff.iloc[i]
        if np.isnan(prev) or np.isnan(cur):
            continue
        if prev <= 0 < cur:
            return ("CE", close.iloc[-1], a_fast.iloc[-1], a_slow.iloc[-1], d.index[i])
        if prev >= 0 > cur:
            return ("PE", close.iloc[-1], a_fast.iloc[-1], a_slow.iloc[-1], d.index[i])
    return None


def scan(interval: str, interval_min: int, fast: int, slow: int, lookback: int, closed_only: bool):
    data = fetch(interval, tuple(FNO_STOCKS))
    ce, pe = [], []
    for sym, df in data.items():
        r = check_cross(df, interval_min, fast, slow, lookback, closed_only)
        if r is None:
            continue
        side, ltp, af, asl, t = r
        row = {
            "Stock": sym,
            "Close": round(float(ltp), 2),
            f"ALMA {fast}": round(float(af), 2),
            f"ALMA {slow}": round(float(asl), 2),
            "Cross Time": t.strftime("%H:%M"),
        }
        (ce if side == "CE" else pe).append(row)
    return pd.DataFrame(ce), pd.DataFrame(pe), len(data)


# ---------------- UI ----------------
st.title("F&O ALMA Screener")
st.caption("CE: ALMA 9 crosses above ALMA 50  |  PE: ALMA 9 crosses below ALMA 50")

with st.sidebar:
    st.header("Settings")
    fast = st.number_input("Fast ALMA length", 2, 200, 9)
    slow = st.number_input("Slow ALMA length", 3, 400, 50)
    lookback = st.slider("Crossover in last N closed candles", 1, 5, 1,
                         help="1 = crossover happened on the latest closed candle only")
    closed_only = st.checkbox("Ignore the still-forming candle", value=True)
    auto = st.checkbox("Auto-refresh every 60 sec", value=False)
    if st.button("Refresh now"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Scanning F&O stocks..."):
    ce2, pe2, n2 = scan("2m", 2, int(fast), int(slow), lookback, closed_only)
    ce5, pe5, n5 = scan("5m", 5, int(fast), int(slow), lookback, closed_only)

st.write(f"Updated: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%d-%b-%Y %H:%M:%S')} IST "
         f"| Data loaded: 2m = {n2}, 5m = {n5} stocks")


def show(title, df):
    st.subheader(f"{title} ({len(df)})")
    if df.empty:
        st.info("कोई stock नहीं मिला")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


tab2, tab5 = st.tabs(["2 Minute", "5 Minute"])
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        show("CE (ALMA 9 ↑ ALMA 50)", ce2)
    with c2:
        show("PE (ALMA 9 ↓ ALMA 50)", pe2)
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        show("CE (ALMA 9 ↑ ALMA 50)", ce5)
    with c2:
        show("PE (ALMA 9 ↓ ALMA 50)", pe5)

if auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
