"""
Streamlit Dashboard - Stock Price Forecasting & Strategy Backtest
Chạy: streamlit run dashboard/app.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.features import build_full_feature_set
from src.backtest import generate_sma_crossover_signals, run_backtest, compare_strategy_vs_buyhold

st.set_page_config(page_title="Stock Forecasting & Backtest", layout="wide", page_icon="📉")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "stock_prices_clean.csv"), parse_dates=["Date"])
    return build_full_feature_set(df)


df = load_data()

st.title("📉 Stock Price Forecasting & Strategy Backtest")
st.caption("Phân tích giá cổ phiếu, dự báo bằng ARIMA/Prophet, và backtest chiến lược SMA Crossover")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Giá hiện tại", f"${df['Close'].iloc[-1]:.2f}")
total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1)
col2.metric("Tổng lợi nhuận (Buy & Hold)", f"{total_return:.1%}")
col3.metric("Volatility hiện tại", f"{df['Volatility'].iloc[-1]:.1%}")
col4.metric("RSI hiện tại", f"{df['RSI'].iloc[-1]:.1f}")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Phân tích giá", "🔮 Dự báo", "⚙️ Backtest chiến lược"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="#94a3b8", width=1)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", line=dict(color="#dc2626")))
    fig.update_layout(title="Giá & Đường trung bình động")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Volatility"], line=dict(color="#dc2626")))
        fig.update_layout(title="Volatility hàng năm hóa (rolling 20 ngày)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], line=dict(color="#7c3aed")))
        fig.add_hline(y=70, line_dash="dash", annotation_text="Quá mua (70)")
        fig.add_hline(y=30, line_dash="dash", annotation_text="Quá bán (30)")
        fig.update_layout(title="RSI (Relative Strength Index)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("So sánh mô hình dự báo trên tập test (30 ngày gần nhất)")

    comparison_path = os.path.join(BASE_DIR, "outputs", "forecast_comparison.csv")
    if os.path.exists(comparison_path):
        forecast_summary = pd.read_csv(comparison_path)
        st.dataframe(forecast_summary.style.highlight_min(subset=["mae", "rmse", "mape"], color="#bbf7d0"),
                     use_container_width=True)
        st.info(
            "💡 Model dự báo PHẢI đánh bại được **Naive Baseline** (dự đoán giá không đổi) mới có giá trị "
            "thực tế — đây là bước sanity-check quan trọng, thường bị bỏ qua khi làm forecasting."
        )
    else:
        st.warning("Chưa có kết quả so sánh forecast. Chạy `python3 -m src.forecasting` trước.")

with tab3:
    st.subheader("Backtest chiến lược SMA Crossover vs Buy & Hold")

    c1, c2 = st.columns(2)
    with c1:
        short_window = st.slider("SMA ngắn hạn", 5, 30, 20)
    with c2:
        long_window = st.slider("SMA dài hạn", 30, 100, 50)

    df_bt = df.dropna(subset=[f"SMA_{20}", f"SMA_{50}"]).copy() if short_window == 20 and long_window == 50 else None

    if df_bt is None:
        # Tính lại SMA theo tham số người dùng chọn
        df_custom = df.copy()
        df_custom[f"SMA_{short_window}"] = df_custom["Close"].rolling(short_window).mean()
        df_custom[f"SMA_{long_window}"] = df_custom["Close"].rolling(long_window).mean()
        df_bt = df_custom.dropna(subset=[f"SMA_{short_window}", f"SMA_{long_window}"]).copy()

    df_signals = generate_sma_crossover_signals(df_bt, short_window=short_window, long_window=long_window)
    df_backtest = run_backtest(df_signals)
    comparison = compare_strategy_vs_buyhold(df_backtest)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_backtest["Date"], y=df_backtest["StrategyEquity"],
                               name="SMA Crossover", line=dict(color="#059669")))
    fig.add_trace(go.Scatter(x=df_backtest["Date"], y=df_backtest["BuyHoldEquity"],
                               name="Buy & Hold", line=dict(color="#94a3b8")))
    fig.update_layout(title="Đường giá trị danh mục (khởi điểm $10,000)")
    st.plotly_chart(fig, use_container_width=True)

    display_comparison = comparison.copy()
    for col in ["total_return", "annualized_return", "annualized_volatility", "max_drawdown", "win_rate"]:
        display_comparison[col] = (display_comparison[col] * 100).round(1).astype(str) + "%"
    display_comparison["sharpe_ratio"] = display_comparison["sharpe_ratio"].round(2)
    st.dataframe(display_comparison, use_container_width=True)

    st.info(
        "💡 **Sharpe Ratio** đo lợi nhuận điều chỉnh theo rủi ro — quan trọng hơn tổng lợi nhuận đơn "
        "thuần, vì chiến lược rủi ro cao có thể có lợi nhuận cao nhưng không bền vững. "
        "**Max Drawdown** đo mức lỗ tối đa từ đỉnh — chỉ số thể hiện chiến lược có 'chịu đựng được' không."
    )
