"""Sinh các biểu đồ EDA và backtest chính."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def plot_price_history(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["Date"], df["Close"], color="#2563eb", linewidth=1)
    ax.set_title("Giá đóng cửa theo thời gian")
    ax.set_ylabel("Giá ($)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_price_with_sma(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["Date"], df["Close"], color="#94a3b8", linewidth=0.8, label="Close", alpha=0.7)
    ax.plot(df["Date"], df["SMA_20"], color="#2563eb", linewidth=1.3, label="SMA 20")
    ax.plot(df["Date"], df["SMA_50"], color="#dc2626", linewidth=1.3, label="SMA 50")
    ax.set_title("Giá & Đường trung bình động (SMA 20/50)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_return_distribution(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["DailyReturn"].dropna().hist(bins=60, ax=ax, color="#7c3aed", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Phân bố lợi nhuận theo ngày (Daily Return)")
    ax.set_xlabel("Daily Return")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_volatility(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["Date"], df["Volatility"], color="#dc2626")
    ax.set_title("Biến động hàng năm hóa (Rolling 20-day Annualized Volatility)")
    ax.set_ylabel("Volatility")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_equity_curve(df_backtest: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df_backtest["Date"], df_backtest["StrategyEquity"], color="#059669", label="SMA Crossover")
    ax.plot(df_backtest["Date"], df_backtest["BuyHoldEquity"], color="#94a3b8", label="Buy & Hold")
    ax.set_title("Đường giá trị danh mục: Chiến lược vs Buy & Hold")
    ax.set_ylabel("Giá trị danh mục ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_forecast_vs_actual(test_dates, y_true, forecasts: dict, out_path: str):
    """forecasts: dict {model_name: predicted_values}"""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_dates, y_true, color="black", linewidth=1.5, label="Giá thực tế", marker="o", markersize=3)
    colors = ["#2563eb", "#dc2626", "#059669"]
    for (name, pred), color in zip(forecasts.items(), colors):
        ax.plot(test_dates, pred, linewidth=1.3, label=name, color=color, linestyle="--")
    ax.set_title("So sánh dự báo vs giá thực tế (tập test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    from src.features import build_full_feature_set

    df = pd.read_csv("data/processed/stock_prices_clean.csv", parse_dates=["Date"])
    df_feat = build_full_feature_set(df)

    plot_price_history(df_feat, "outputs/price_history.png")
    plot_price_with_sma(df_feat, "outputs/price_with_sma.png")
    plot_return_distribution(df_feat, "outputs/return_distribution.png")
    plot_volatility(df_feat, "outputs/volatility.png")
    print("Đã lưu 4 biểu đồ EDA vào outputs/")
