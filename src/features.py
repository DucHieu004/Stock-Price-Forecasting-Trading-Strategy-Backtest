"""
Module tính các chỉ báo kỹ thuật (technical indicators) chuẩn dùng trong phân
tích tài chính - làm input cho cả forecasting và backtest chiến lược.

- Daily Return: lợi nhuận theo ngày = (Close_t / Close_t-1) - 1
- SMA (Simple Moving Average): trung bình động - làm mượt nhiễu ngắn hạn để
  thấy xu hướng, dùng cho chiến lược crossover
- Volatility (rolling std của return): đo mức độ biến động - input quan
  trọng cho quản lý rủi ro
- RSI (Relative Strength Index): đo momentum, dao động 0-100, thường dùng để
  xác định vùng quá mua (>70) / quá bán (<30)
- Log Return: dùng cho ARIMA vì có tính chất thống kê ổn định (stationary)
  hơn giá gốc - giá cổ phiếu thường KHÔNG stationary (có trend), nhưng log
  return thường gần stationary hơn, phù hợp giả định của ARIMA
"""
import pandas as pd
import numpy as np


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DailyReturn"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_moving_averages(df: pd.DataFrame, windows: list = [20, 50]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volatility hàng năm hóa (annualized) từ độ lệch chuẩn của return theo
    ngày - nhân với sqrt(252) vì có ~252 ngày giao dịch/năm, đây là quy ước
    chuẩn để so sánh volatility giữa các khung thời gian khác nhau."""
    df = df.copy()
    df["Volatility"] = df["DailyReturn"].rolling(window=window).std() * np.sqrt(252)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def build_full_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    df = add_returns(df)
    df = add_moving_averages(df, windows=[20, 50])
    df = add_volatility(df)
    df = add_rsi(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/stock_prices_clean.csv", parse_dates=["Date"])
    df_feat = build_full_feature_set(df)
    print(df_feat[["Date", "Close", "DailyReturn", "SMA_20", "SMA_50", "Volatility", "RSI"]].tail(10))
    df_feat.to_csv("data/processed/stock_prices_features.csv", index=False)
    print("\nĐã lưu -> data/processed/stock_prices_features.csv")
