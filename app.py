# app.py
# Streamlit skeleton for Quasi-Passive Portfolio Project
# Run with: streamlit run app.py

import importlib

# Load Streamlit dynamically so static analyzers do not require the optional
# runtime dependency to be installed while inspecting this module.
st = importlib.import_module("streamlit")
pd = importlib.import_module("pandas")
np = importlib.import_module("numpy")
yf = importlib.import_module("yfinance")
from datetime import datetime, timedelta
import os
import pickle

# -----------------------
# Configuration / Globals
# -----------------------
TICKERS = [
    "SPY","VTI","IJR","IWM","VTV","VUG","MTUM","QUAL","SCHD","USMV",
    "VEA","EFA","EFV","VWO",
    "BND","BNDX","TIP","SHV","SHY","IEF","TLT","LQD","HYG","MUB","EMBD",
    "VNQ","VNQI","DBC","GLD","XLE",
    "XLK","XLF","SPLV","VIG","SCHP",
    "SPYU","TQQQ","SPXU","TMF","TBF"
]

LEVERAGED_TICKERS = {"SPYU","TQQQ","SPXU","TMF","TBF"}

CACHE_DIR = ".cache_streamlit"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "price_cache.pkl")

DEFAULT_START = "1999-01-01"
TODAY = datetime.utcnow().date().isoformat()

# -----------------------
# Utility functions
# -----------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date=DEFAULT_START):
    # Use yfinance for adjusted close prices; return DataFrame of daily adj close
    df = yf.download(tickers, start=start_date, progress=False, auto_adjust=False)
    # sometimes yf returns single-column series if single ticker; keep robust
    if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
        prices = df["Adj Close"].copy()
    else:
        # If yf.version changes, try the common expected structure
        prices = df
    prices = prices.sort_index()
    return prices

def load_cached_prices():
    # Tries to load cache, else downloads and saves cache.
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception:
            pass
    # download and cache
    data = download_prices(TICKERS, DEFAULT_START)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    return data

def simple_momentum(prices, lookback_days=252):
    # returns pct change over lookback_days for each column aligned to last available date
    return prices.pct_change(lookback_days).iloc[-1]

def simple_value_proxy(prices):
    # placeholder: could use inverse volatility, dividend yield proxies, or fundamental CSV
    # For now, use 1 / 252-day volatility as a 'value' proxy (higher is more 'value')
    returns = prices.pct_change().dropna(how="all")
    vol = returns.rolling(window=252).std().iloc[-1]
    proxy = 1 / vol
    return proxy

def get_glidepath_allocation(horizon_years):
    # Very simple linear glidepath example (customize as desired)
    # Equity weight = min(max(0.5 + 0.02*horizon_years, 0), 1)
    # For demonstration, map horizon:
    if horizon_years >= 20:
        eq = 0.9
    elif horizon_years >= 15:
        eq = 0.8
    elif horizon_years >= 5:
        eq = 0.6
    else:
        # retirement or immediate: preservation
        eq = 0.35
    return eq

def build_naive_allocation(universe, equity_weight, factor_scores=None):
    # Simple allocation: split equity_weight equally across equity tickers,
    # rest to fixed income tickers. Factor_scores (momentum/value) can be used to tilt.
    # Identify buckets by naive rules
    equities = [t for t in universe if t in {
        "SPY","VTI","IJR","IWM","VTV","VUG","MTUM","QUAL","SCHD","USMV",
        "VEA","EFA","EFV","VWO","XLK","XLF","SPLV","VIG","SCHP"
    }]
    fixed = [t for t in universe if t in {
        "BND","BNDX","TIP","SHV","SHY","IEF","TLT","LQD","HYG","MUB","EMBD"
    }]
    realassets = [t for t in universe if t in {"VNQ","VNQI","DBC","GLD","XLE"}]
    others = list(set(universe) - set(equities) - set(fixed) - set(realassets))
    # For demo: allocate equities portion across equities + factor/sector/other equities,
    # remainder to fixed income and a small allocation to real assets
    alloc = pd.Series(0.0, index=universe)
    if len(equities) > 0:
        eq_alloc_each = equity_weight / len(equities)
        alloc.loc[equities] = eq_alloc_each
    # place 10% of portfolio into real assets if available (as part of non-equity)
    ra_share = 0.10 * (1 - equity_weight)
    fixed_share = (1 - equity_weight) - ra_share
    if len(realassets) > 0:
        ra_alloc_each = ra_share / len(realassets)
        alloc.loc[realassets] = ra_alloc_each
    if len(fixed) > 0:
        fixed_alloc_each = fixed_share / len(fixed)
        alloc.loc[fixed] = fixed_alloc_each
    # Put any remaining small residual into "others"
    residual = 1.0 - alloc.sum()
    if residual > 0 and len(others) > 0:
        alloc.loc[others] = residual / len(others)
    # Enforce bounds 0% to 12% per spec
    alloc = alloc.clip(lower=0.0, upper=0.12)
    # Normalize to full 1.0 after clipping
    if alloc.sum() > 0:
        alloc = alloc / alloc.sum()
    return alloc

def flag_leveraged(alloc):
    # returns list of leveraged tickers present with non-zero allocation
    return [t for t in alloc.index if t in LEVERAGED_TICKERS and alloc.loc[t] > 0]

# -----------------------
# Streamlit App Layout
# -----------------------
st.set_page_config(page_title="Quasi-Passive Portfolio - Starter UI", layout="wide")
st.title("Quasi-Passive Portfolio — Starter Streamlit App")

# Sidebar controls
st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(DEFAULT_START))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(TODAY))
horizon = st.sidebar.selectbox("Investment Horizon", ("20 years", "15 years", "5 years", "Retirement"))
run_button = st.sidebar.button("Run Starter Backtest")

st.sidebar.markdown("Rebalancing: Twice-monthly (1st & 18th) — Quarterly re-optimization (placeholder)")
st.sidebar.markdown("Transaction cost: 5 bps per trade")

st.sidebar.header("Universe")
st.sidebar.write(f"{len(TICKERS)} tickers selected")
st.sidebar.write(", ".join(TICKERS))

# Main content: data load + quick stats
st.header("Data & Universe Check")
with st.spinner("Loading cached price data..."):
    prices = load_cached_prices()
if prices is None or prices.empty:
    st.error("No price data available. Check internet connection or yfinance.")
    st.stop()

# Show data timeframe
st.write("Price data range detected:")
st.write(f"From {prices.index.min().date()} to {prices.index.max().date()} (UTC)")

# Quick table of latest prices and presence
latest = prices.ffill().iloc[-1].rename("Last Price")
df_summary = pd.DataFrame({
    "Last Price": latest,
    "Data Points": prices.notna().sum()
})
st.dataframe(df_summary.sort_values("Data Points", ascending=False).head(20))

# Factor calculations (basic placeholders)
st.header("Factor Proxies")
with st.expander("Factor calculation details"):
    st.write("Momentum: 252-day total return (placeholder)")
    st.write("Value proxy: inverse of 252-day volatility (placeholder)")

momentum_scores = simple_momentum(prices)
value_scores = simple_value_proxy(prices)

factor_df = pd.DataFrame({
    "Momentum": momentum_scores,
    "ValueProxy": value_scores
}).sort_values("Momentum", ascending=False).head(30)

st.dataframe(factor_df)

# Allocation Generation
st.header("Glidepath & Allocation (Starter)")

horizon_map = {"20 years": 20, "15 years": 15, "5 years": 5, "Retirement": 0}
yrs = horizon_map[horizon]
equity_weight = get_glidepath_allocation(yrs)
st.write(f"Selected horizon: {horizon} → target equity weight = {equity_weight:.0%}")

alloc = build_naive_allocation(TICKERS, equity_weight, factor_scores=momentum_scores)
st.subheader("Starter Allocation (naive / placeholder)")
st.table(alloc[alloc > 0].sort_values(ascending=False).round(4).to_frame("Weight"))

levs = flag_leveraged(alloc)
if levs:
    st.warning(f"Leveraged/Inverse instruments present in allocation: {', '.join(levs)}")
    st.info("These instruments are modeled at daily frequency and may exhibit volatility drag / path-dependence. Use with caution.")

# Placeholder backtest run
st.header("Starter Backtest (placeholder)")
st.markdown("This section runs a very simple backtest to show how allocations behave. Replace with your walk-forward backtest & CVXPY optimizer.")
if run_button:
    # Simple portfolio daily returns simulation (naive, no rebalancing)
    st.info("Running starter backtest (naive, no rebalancing). Replace with walk-forward engine.")
    # Align prices, compute returns
    aligned = prices[TICKERS].ffill().dropna(how="all")
    rets = aligned.pct_change().fillna(0)
    # use last allocation (weights on most recent date) to create naive daily portfolio returns
    w = alloc.reindex(rets.columns).fillna(0).values  # ensure same order
    port_rets = rets.dot(w)
    port_cum = (1 + port_rets).cumprod()
    benchmark = rets["SPY"] if "SPY" in rets.columns else rets.iloc[:, 0]
    bench_cum = (1 + benchmark).cumprod()

    # Simple metrics
    total_ret = port_cum.iloc[-1] - 1
    cagr = (port_cum.iloc[-1]) ** (252.0 / len(port_rets)) - 1 if len(port_rets) > 0 else np.nan
    vol = port_rets.std() * np.sqrt(252)
    sharpe = (port_rets.mean() * 252) / (port_rets.std() * np.sqrt(252)) if port_rets.std() > 0 else np.nan
    max_dd = (port_cum.cummax() - port_cum).max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_ret:.2%}")
    col2.metric("CAGR (ann.)", f"{cagr:.2%}")
    col3.metric("Vol (ann.)", f"{vol:.2%}")
    col4.metric("Sharpe (ann.)", f"{sharpe:.2f}")

    st.line_chart(pd.DataFrame({
        "Strategy": port_cum,
        "Benchmark (SPY)": bench_cum
    }).dropna())

    st.markdown("Notes: This backtest is purely illustrative. Replace with: quarterly re-optimization, twice-monthly calendar rebalancing, trade costs (5 bps), min trade threshold, and walk-forward validation.")

# Placeholders / Actions
st.sidebar.header("Next actions")
st.sidebar.button("Run Walk-Forward Backtest (placeholder)")
if st.sidebar.button("Replace with CVXPY optimizer"):
    st.info("Triggering optimizer replacement is not implemented in this skeleton — add your CVXPY function and call it here.")

st.sidebar.markdown("""
To implement next:
- Replace build_naive_allocation with CVXPY optimizer using 10-year rolling lookback.
- Implement rebalancing schedule (1st & 18th) and apply transaction costs and trade thresholds.
- Add tax-drag model and turnover reporting.
- Add UI panels for parameter sweeps (e.g., factor tilt weights, max position bounds).
""")

st.caption("Starter Streamlit app — extend with your optimization and walk-forward backtest modules.")
