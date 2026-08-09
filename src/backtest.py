"""
Module backtest chiến lược giao dịch dựa trên SMA Crossover - 1 trong những
chiến lược kinh điển nhất để bắt đầu học quant trading.

Chiến lược SMA Crossover:
- Khi SMA ngắn hạn (VD: 20 ngày) CẮT LÊN TRÊN SMA dài hạn (VD: 50 ngày)
  -> tín hiệu MUA (xu hướng tăng đang hình thành - "Golden Cross")
- Khi SMA ngắn hạn CẮT XUỐNG DƯỚI SMA dài hạn
  -> tín hiệu BÁN (xu hướng giảm đang hình thành - "Death Cross")

QUAN TRỌNG - các nguyên tắc backtest đúng cách (nhiều người mới làm sai):
1. Tín hiệu tính bằng dữ liệu NGÀY t, nhưng phải THỰC HIỆN GIAO DỊCH ở ngày
   t+1 (không thể mua/bán ngay trong ngày phát sinh tín hiệu bằng giá đóng
   cửa của chính ngày đó - đây gọi là "look-ahead bias", lỗi rất phổ biến)
2. Phải trừ chi phí giao dịch (transaction cost) - nếu không, chiến lược có
   vẻ sinh lời nhưng thực tế lỗ vì phí giao dịch ăn hết lợi nhuận
3. Đánh giá bằng Sharpe Ratio (lợi nhuận điều chỉnh theo rủi ro), Max
   Drawdown (mức lỗ tối đa từ đỉnh), không chỉ nhìn tổng lợi nhuận
"""
import numpy as np
import pandas as pd


def generate_sma_crossover_signals(df: pd.DataFrame, short_window: int = 20,
                                      long_window: int = 50) -> pd.DataFrame:
    """Sinh tín hiệu giao dịch: 1 = giữ vị thế mua (long), 0 = không giữ vị thế.
    Dùng .shift(1) để đảm bảo tín hiệu tính từ dữ liệu QUÁ KHỨ, tránh
    look-ahead bias."""
    df = df.copy()
    df["Signal"] = 0
    # Golden Cross: SMA ngắn > SMA dài -> giữ vị thế mua
    df.loc[df[f"SMA_{short_window}"] > df[f"SMA_{long_window}"], "Signal"] = 1
    # QUAN TRỌNG: shift(1) để tín hiệu ngày t chỉ dùng được để giao dịch ở
    # ngày t+1 - mô phỏng đúng thực tế (không thể biết giá đóng cửa hôm nay
    # trước khi thị trường đóng cửa)
    df["Position"] = df["Signal"].shift(1).fillna(0)
    return df


def run_backtest(df: pd.DataFrame, transaction_cost: float = 0.001,
                   initial_capital: float = 10000) -> pd.DataFrame:
    """Chạy backtest đầy đủ: tính lợi nhuận chiến lược sau khi trừ phí giao
    dịch, so sánh với chiến lược Buy & Hold (mua giữ, không giao dịch gì thêm)."""
    df = df.copy()

    # Lợi nhuận chiến lược = lợi nhuận thị trường ngày đó * có đang giữ vị thế không
    df["StrategyReturn"] = df["DailyReturn"] * df["Position"]

    # Trừ phí giao dịch mỗi khi ĐỔI vị thế (mua mới hoặc bán ra)
    df["PositionChange"] = df["Position"].diff().abs().fillna(0)
    df["TransactionCost"] = df["PositionChange"] * transaction_cost
    df["StrategyReturnNet"] = df["StrategyReturn"] - df["TransactionCost"]

    # Tính giá trị danh mục lũy kế
    df["StrategyEquity"] = initial_capital * (1 + df["StrategyReturnNet"]).cumprod()
    df["BuyHoldEquity"] = initial_capital * (1 + df["DailyReturn"].fillna(0)).cumprod()

    return df


def compute_performance_metrics(df: pd.DataFrame, return_col: str = "StrategyReturnNet") -> dict:
    """Tính các chỉ số hiệu năng chuẩn ngành quant để đánh giá chiến lược."""
    returns = df[return_col].dropna()

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annualized_vol = returns.std() * np.sqrt(252)

    # Sharpe Ratio: lợi nhuận vượt trội trên mỗi đơn vị rủi ro (giả định
    # risk-free rate = 0 để đơn giản hóa - trong thực tế nên trừ lãi suất
    # phi rủi ro, VD lãi suất trái phiếu chính phủ)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max Drawdown: mức sụt giảm tối đa từ đỉnh - chỉ số quan trọng đo rủi ro
    # "chịu đựng" được của chiến lược, không chỉ nhìn lợi nhuận trung bình
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def compare_strategy_vs_buyhold(df: pd.DataFrame) -> pd.DataFrame:
    strategy_metrics = compute_performance_metrics(df, return_col="StrategyReturnNet")
    buyhold_metrics = compute_performance_metrics(df.assign(DailyReturn=df["DailyReturn"]),
                                                     return_col="DailyReturn")

    comparison = pd.DataFrame([
        {"Strategy": "SMA Crossover", **strategy_metrics},
        {"Strategy": "Buy & Hold", **buyhold_metrics},
    ])
    return comparison


if __name__ == "__main__":
    df = pd.read_csv("data/processed/stock_prices_features.csv", parse_dates=["Date"])
    df = df.dropna(subset=["SMA_20", "SMA_50"]).reset_index(drop=True)

    df_signals = generate_sma_crossover_signals(df, short_window=20, long_window=50)
    df_backtest = run_backtest(df_signals)

    n_trades = (df_backtest["PositionChange"] > 0).sum()
    print(f"Số lần đổi vị thế (giao dịch): {n_trades}")

    comparison = compare_strategy_vs_buyhold(df_backtest)
    print("\n=== So sánh chiến lược SMA Crossover vs Buy & Hold ===")
    print(comparison.to_string(index=False))

    comparison.to_csv("outputs/strategy_comparison.csv", index=False)
    df_backtest.to_csv("data/processed/backtest_results.csv", index=False)
    print("\nĐã lưu -> outputs/strategy_comparison.csv, data/processed/backtest_results.csv")
