"""
Tạo dữ liệu MẪU (synthetic) mô phỏng giá cổ phiếu OHLCV (Open/High/Low/Close/
Volume) theo ngày, đúng cấu trúc dữ liệu lấy từ Yahoo Finance / các API giá
chứng khoán phổ biến.

Mô phỏng 2 đặc điểm quan trọng của giá cổ phiếu THẬT (không chỉ random walk
đơn thuần):
1. Volatility clustering: giai đoạn biến động mạnh có xu hướng đi liền nhau
   (GARCH-like effect) - đặc điểm nổi tiếng của dữ liệu tài chính thật
2. Trend + mean-reversion pha trộn: giá có xu hướng dài hạn nhưng dao động
   quanh đường trend trong ngắn hạn

Cách thay bằng dữ liệu thật:
  1. Cài `yfinance`: pip install yfinance
  2. Tải dữ liệu: `yf.download("AAPL", start="2022-01-01", end="2024-01-01")`
  3. Đặt vào data/raw/stock_prices.csv với cùng cấu trúc cột
     (Date, Open, High, Low, Close, Volume)
"""
import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)

TICKER = "DEMO"
N_DAYS = 750  # ~3 năm giao dịch (250 ngày/năm)
START_PRICE = 150.0
start_date = pd.Timestamp("2022-01-03")

# Mô phỏng volatility clustering bằng quá trình GARCH(1,1) đơn giản hóa
# (tham số được chọn để tổng alpha+beta < 1 đủ xa, tránh persistence quá mức
# gây hiệu ứng khuếch đại phi thực tế khi kết hợp với shock đuôi dày)
omega, alpha, beta = 0.00002, 0.05, 0.90
variance = np.zeros(N_DAYS)
variance[0] = 0.0004  # variance khởi điểm (~2% daily vol)
returns = np.zeros(N_DAYS)
MAX_DAILY_VOL = 0.06  # giới hạn an toàn: không cho variance nổ quá mức thực tế

# Thêm 1 xu hướng tăng dài hạn nhẹ + vài "regime" biến động khác nhau để
# giống thị trường thật (giai đoạn tăng ổn định, giai đoạn biến động mạnh)
drift = 0.0004
for t in range(1, N_DAYS):
    variance[t] = min(omega + alpha * returns[t-1]**2 + beta * variance[t-1], MAX_DAILY_VOL**2)
    shock = np.random.standard_t(df=6) * np.sqrt(variance[t])
    # Trừ nửa variance (Itô correction) để drift trung bình đúng như kỳ vọng,
    # tránh giá bị lệch âm tích lũy do biến động cao
    returns[t] = np.clip(drift - 0.5 * variance[t] + shock, -0.15, 0.15)

# Thêm 1 giai đoạn "sự kiện thị trường" (market shock) ở khoảng giữa dữ liệu
# để notebook có gì đó thú vị để phân tích (giống 1 đợt điều chỉnh thị trường thật)
shock_start, shock_end = 400, 430
returns[shock_start:shock_end] -= 0.015  # giảm mạnh trong ~1 tháng
returns[shock_end:shock_end+20] += 0.008  # hồi phục sau đó

prices = START_PRICE * np.exp(np.cumsum(returns))

dates = pd.bdate_range(start=start_date, periods=N_DAYS)  # chỉ ngày giao dịch (bỏ cuối tuần)

# Sinh Open/High/Low từ Close với biến động trong ngày hợp lý
daily_range = np.abs(np.random.normal(0, 1, N_DAYS)) * prices * np.sqrt(variance) * 1.5
open_prices = prices * (1 + np.random.normal(0, 0.003, N_DAYS))
high_prices = np.maximum(open_prices, prices) + daily_range * np.random.uniform(0.3, 0.7, N_DAYS)
low_prices = np.minimum(open_prices, prices) - daily_range * np.random.uniform(0.3, 0.7, N_DAYS)

# Volume tương quan với biến động (ngày biến động mạnh -> volume cao hơn, giống thực tế)
base_volume = 5_000_000
volume = (base_volume * (1 + 8 * np.sqrt(variance)) * np.random.uniform(0.7, 1.3, N_DAYS)).astype(int)

df = pd.DataFrame({
    "Date": dates,
    "Open": np.round(open_prices, 2),
    "High": np.round(high_prices, 2),
    "Low": np.round(low_prices, 2),
    "Close": np.round(prices, 2),
    "Volume": volume,
})

df.to_csv("data/raw/stock_prices.csv", index=False)
print(f"Đã tạo {len(df):,} ngày dữ liệu giá ({TICKER}) -> data/raw/stock_prices.csv")
print(f"Khoảng thời gian: {df['Date'].min().date()} -> {df['Date'].max().date()}")
print(f"Giá đầu: ${df['Close'].iloc[0]:.2f} | Giá cuối: ${df['Close'].iloc[-1]:.2f}")
print(df.head())
