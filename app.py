# app.py
# Streamlit yfinance Dashboard
# pip install streamlit yfinance pandas numpy matplotlib seaborn plotly openpyxl scipy

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy import stats
import os

st.set_page_config(page_title="yfinance Dashboard", layout="wide")

# --------------------------
# Constants & Utility
# --------------------------
TRADING_DAYS = 252

def annualized_mean(daily_return):
    return daily_return.mean() * TRADING_DAYS

def annualized_vol(daily_return):
    return daily_return.std() * np.sqrt(TRADING_DAYS)

def downside_deviation(daily_return, mar=0.0):
    neg = daily_return[daily_return < mar]
    if len(neg)==0: return 0.0
    return np.sqrt(((neg - mar)**2).mean()) * np.sqrt(TRADING_DAYS)

def sortino_ratio(daily_return, mar=0.0):
    dd = downside_deviation(daily_return, mar)
    if dd == 0: return np.nan
    return (annualized_mean(daily_return) - mar)/dd

def calculate_max_drawdown_pct(daily_return):
    cum = (1+daily_return).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max)/running_max
    return drawdown.min()

def calmar_ratio(daily_return):
    ann = annualized_mean(daily_return)
    dd = calculate_max_drawdown_pct(daily_return)
    if dd == 0: return np.nan
    return ann / abs(dd)

def omega_ratio(daily_return, threshold=0.0):
    excess = daily_return - threshold
    pos = excess[excess>0].sum()
    neg = -excess[excess<0].sum()
    if neg == 0: return np.nan
    return pos / neg

def var_cvar(series, alpha=0.05):
    q = series.quantile(alpha)
    cvar = series[series <= q].mean()
    return q, cvar

def upside_downside_capture(benchmark_returns, asset_returns):
    df = pd.concat([benchmark_returns, asset_returns], axis=1).dropna()
    if df.shape[0]==0: return np.nan, np.nan
    bm = df.iloc[:,0]
    at = df.iloc[:,1]
    up_idx = bm > 0
    down_idx = bm < 0
    upside = ((1+at[up_idx]).prod()-1)/((1+bm[up_idx]).prod()-1) if up_idx.sum()>0 and ((1+bm[up_idx]).prod()-1)!=0 else np.nan
    downside = ((1+at[down_idx]).prod()-1)/((1+bm[down_idx]).prod()-1) if down_idx.sum()>0 and ((1+bm[down_idx]).prod()-1)!=0 else np.nan
    return upside, downside

# --------------------------
# Analyzer Class
# --------------------------
class PolishedAnalyzer:
    def __init__(self):
        self.raw = None
        self.prices = None
        self.returns = None
        self.tickers = []
        self.metrics = {}
        self.benchmark = "SPY"
        self.bench_returns = None
        self.rolling_stats = {}

    def fetch(self, tickers, start, end, interval='1d', include_benchmark=True):
        self.tickers = tickers
        to_download = tickers.copy()
        if include_benchmark and self.benchmark not in to_download:
            to_download.append(self.benchmark)
        df = yf.download(to_download, start=start, end=end + pd.Timedelta(days=1),
                         interval=interval, auto_adjust=True, threads=True, progress=False, group_by='ticker')
        # build prices
        if isinstance(df.columns, pd.MultiIndex):
            close = {}
            for t in to_download:
                try:
                    close[t] = df[(t, 'Close')]
                except:
                    close[t] = df[(t, 'Adj Close')]
            prices = pd.DataFrame(close)
        else:
            prices = pd.DataFrame({to_download[0]: df['Close']}) if len(to_download)==1 else pd.DataFrame(df['Close'])
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().dropna(how='all', axis=1)
        # separate benchmark
        if include_benchmark and self.benchmark in prices.columns:
            self.bench_returns = prices[self.benchmark].pct_change().dropna()
            prices = prices.drop(columns=[self.benchmark])
        else:
            self.bench_returns = None
        self.raw = df
        self.prices = prices
        self.returns = prices.pct_change().dropna(how='all')
        if self.returns.empty:
            raise ValueError("Not enough data to compute returns.")
        self._compute_metrics()
        self._compute_rolling_stats()

    def _compute_metrics(self):
        self.metrics = {}
        for t in self.returns.columns:
            r = self.returns[t].dropna()
            if len(r) < 2:
                continue
            total = (1+r).prod() - 1
            ann = annualized_mean(r)
            vol = annualized_vol(r)
            sharpe = ann / vol if vol>0 else np.nan
            maxdd = calculate_max_drawdown_pct(r)
            var95, cvar95 = var_cvar(r,0.05)
            skew = r.skew()
            kurt = r.kurtosis()
            down_dev = downside_deviation(r)
            s_ratio = sortino_ratio(r)
            calmar = calmar_ratio(r)
            omega = omega_ratio(r)
            upside_cap, downside_cap = (np.nan, np.nan)
            if self.bench_returns is not None:
                upside_cap, downside_cap = upside_downside_capture(self.bench_returns.loc[r.index.intersection(self.bench_returns.index)], r)
            self.metrics[t] = {
                'Total Return': total,
                'Annual Return': ann,
                'Volatility': vol,
                'Sharpe': sharpe,
                'Sortino': s_ratio,
                'Downside Dev': down_dev,
                'Calmar': calmar,
                'Omega': omega,
                'Max Drawdown': maxdd,
                'VaR 95%': var95,
                'CVaR 95%': cvar95,
                'Skewness': skew,
                'Kurtosis': kurt,
                'Upside Capture vs SPY': upside_cap,
                'Downside Capture vs SPY': downside_cap
            }

    def _compute_rolling_stats(self, windows=[21,63,126]):
        self.rolling_stats = {}
        for t in self.returns.columns:
            r = self.returns[t].dropna()
            self.rolling_stats[t] = {}
            for w in windows:
                self.rolling_stats[t][f'vol_{w}'] = r.rolling(window=w).std()*np.sqrt(TRADING_DAYS)
                self.rolling_stats[t][f'sharpe_{w}'] = r.rolling(window=w).mean()*TRADING_DAYS/(r.rolling(window=w).std()*np.sqrt(TRADING_DAYS))

    # --------- Tables ---------
    def snapshot_table(self):
        snap = []
        today = self.prices.index.max()
        for t in self.prices.columns:
            last = self.prices[t].iloc[-1]
            r1m = (self.prices[t].iloc[-1]/self.prices[t].iloc[-21]-1) if len(self.prices[t])>21 else np.nan
            r3m = (self.prices[t].iloc[-1]/self.prices[t].iloc[-63]-1) if len(self.prices[t])>63 else np.nan
            ytd_start = pd.Timestamp(year=today.year, month=1, day=1)
            r_ytd = (self.prices[t].loc[self.prices.index>=ytd_start].iloc[-1]/self.prices[t].loc[self.prices.index>=ytd_start].iloc[0]-1) if any(self.prices.index>=ytd_start) else np.nan
            r1y = (self.prices[t].iloc[-1]/self.prices[t].iloc[-min(len(self.prices[t]),252)]-1) if len(self.prices[t])>252 else np.nan
            vol = annualized_vol(self.returns[t])
            maxdd = calculate_max_drawdown_pct(self.returns[t])
            snap.append([t, last, r1m, r3m, r_ytd, r1y, vol, maxdd])
        cols = ['Ticker','Last Price','1M','3M','YTD','1Y','Volatility','Max Drawdown']
        snapdf = pd.DataFrame(snap, columns=cols).set_index('Ticker')
        pct_cols = ['1M', '3M', 'YTD', '1Y', 'Volatility', 'Max Drawdown']
        for col in pct_cols:
            snapdf[col] = snapdf[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else '—')
        return snapdf

    def metrics_table(self):
        df = pd.DataFrame(self.metrics).T
        return df

    def plot_normalized(self):
        data = (1+self.returns).cumprod()*100
        data.iloc[0] = 100
        fig, ax = plt.subplots(figsize=(12,6))
        for col in data.columns:
            ax.plot(data.index, data[col], label=col, linewidth=2)
        ax.set_title('Performance (Rebased to 100)')
        ax.set_ylabel('Index (Base 100)')
        ax.legend(bbox_to_anchor=(1.02,1))
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    def plot_correlation(self):
        corr = self.returns.corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, cmap='RdBu_r', center=0, annot=len(corr)<=12, fmt='.2f', ax=ax, square=True, cbar_kws={'shrink':0.8})
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        return fig

    def plot_risk_return(self):
        df = pd.DataFrame(self.metrics).T
        fig, ax = plt.subplots(figsize=(10,7))
        sc = ax.scatter(df['Volatility']*100, df['Annual Return']*100, c=df['Sharpe'], cmap='RdYlGn', s=140, edgecolor='k')
        for i, idx in enumerate(df.index):
            ax.annotate(idx, (df['Volatility'].iloc[i]*100, df['Annual Return'].iloc[i]*100), textcoords='offset points', xytext=(4,4), fontsize=9)
        ax.set_xlabel('Volatility (%)'); ax.set_ylabel('Annual Return (%)'); ax.set_title('Risk-Return Scatter')
        plt.colorbar(sc, label='Sharpe Ratio', shrink=0.7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_rolling_stats(self, tickers=None, windows=[21,63,126]):
        if tickers is None:
            tickers = list(self.returns.columns[:8])
        fig, axs = plt.subplots(2,1,figsize=(12,8), sharex=True)
        for t in tickers:
            rs = self.rolling_stats[t]
            for w in windows:
                if f"vol_{w}" in rs:
                    axs[0].plot(rs[f"vol_{w}"], label=f"{t} vol({w})")
                if f"sharpe_{w}" in rs:
                    axs[1].plot(rs[f"sharpe_{w}"], label=f"{t} sharpe({w})")
        axs[0].set_title("Rolling Volatility (annualized)")
        axs[0].set_ylabel("Volatility")
        axs[1].set_title("Rolling Sharpe Ratio")
        axs[1].set_ylabel("Sharpe")
        axs[1].set_xlabel("Date")
        axs[0].legend(ncol=3, bbox_to_anchor=(1.02,1))
        axs[1].legend(ncol=3, bbox_to_anchor=(1.02,1))
        plt.tight_layout()
        return fig

    def interactive_dashboard(self, max_lines=8):
        data = (1+self.returns).cumprod()*100
        data.iloc[0]=100
        cols = data.columns[:max_lines]
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Perf (rebased)','Risk-Return','Correlation','Returns Dist'),
            specs=[[{"type":"scatter"},{"type":"scatter"}],[{"type":"heatmap"},{"type":"histogram"}]]
        )
        for c in cols:
            fig.add_trace(go.Scatter(x=data.index, y=data[c], name=str(c)), row=1, col=1)
        df = pd.DataFrame(self.metrics).T
        fig.add_trace(go.Scatter(
            x=df['Volatility']*100, y=df['Annual Return']*100,
            mode='markers+text', text=df.index, textposition='top center'
        ), row=1, col=2)
        corr = self.returns.corr()
        fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0), row=2, col=1)
        fig.add_trace(go.Histogram(x=self.returns.values.flatten(), nbinsx=80), row=2, col=2)
        fig.update_layout(height=900, title_text='Interactive Analytics Dashboard')
        return fig

    def export_to_excel(self, path):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            self.prices.to_excel(writer, sheet_name='Prices')
            self.returns.to_excel(writer, sheet_name='Returns')
            self.returns.corr().to_excel(writer, sheet_name='Correlation')
            pd.DataFrame(self.metrics).T.to_excel(writer, sheet_name='Metrics')
            # Rolling stats
            roll_frames = {}
            for t in self.rolling_stats:
                for k,v in self.rolling_stats[t].items():
                    roll_frames[f"{t}_{k}"] = v
            if roll_frames:
                pd.DataFrame(roll_frames).to_excel(writer, sheet_name='RollingStats')
            # Snapshot
            snapdf = self.snapshot_table()
            snapdf.to_excel(writer, sheet_name='Snapshot')
        return path

# --------------------------
# Streamlit UI
# --------------------------
st.title("yfinance Dashboard")

# Inputs
tickers = st.text_input("Tickers (space or comma separated)", "AAPL MSFT NVDA GOOGL AMZN META TSLA")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.Timestamp.today())
freq = st.selectbox("Frequency", ['1d','1wk','1mo'])
selected_analysis = st.multiselect(
    "Analysis",
    ["Price Charts","Snapshot","Metrics","Correlation","Risk-Return","Rolling Stats","Interactive"],
    default=["Price Charts","Snapshot","Metrics"]
)
save_file = st.checkbox("Save to Excel", True)

if st.button("Run Analysis"):
    an = PolishedAnalyzer()
    tickers_list = [t.strip().upper() for t in tickers.replace(',',' ').split() if t.strip()]
    with st.spinner("Fetching data..."):
        an.fetch(tickers_list, pd.to_datetime(start_date), pd.to_datetime(end_date), interval=freq)
    st.success(f"Data fetched for {', '.join(an.prices.columns)} ({len(an.prices)} observations)")

    # Snapshot
    if "Snapshot" in selected_analysis:
        st.subheader("Ticker Snapshot")
        st.dataframe(an.snapshot_table())

    # Metrics
    if "Metrics" in selected_analysis:
        st.subheader("Performance Metrics")
        st.dataframe(an.metrics_table())

    # Price charts
    if "Price Charts" in selected_analysis:
        st.subheader("Price Charts (Normalized)")
        st.pyplot(an.plot_normalized())

    # Correlation
    if "Correlation" in selected_analysis:
        st.subheader("Correlation Matrix")
        st.pyplot(an.plot_correlation())

    # Risk-Return
    if "Risk-Return" in selected_analysis:
        st.subheader("Risk-Return Scatter")
        st.pyplot(an.plot_risk_return())

    # Rolling Stats
    if "Rolling Stats" in selected_analysis:
        st.subheader("Rolling Volatility & Sharpe")
        st.pyplot(an.plot_rolling_stats())

    # Interactive dashboard
    if "Interactive" in selected_analysis:
        st.subheader("Interactive Dashboard")
        st.plotly_chart(an.interactive_dashboard())

    # Export
    if save_file:
        fname = f"analysis_{len(an.prices.columns)}tickers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        path = os.path.join(os.getcwd(), fname)
        an.export_to_excel(path)
        st.success(f"Exported results to {path}")