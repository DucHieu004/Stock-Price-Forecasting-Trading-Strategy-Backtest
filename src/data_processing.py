"""
Module xử lý dữ liệu giá cổ phiếu (OHLCV).

Vấn đề dữ liệu thực tế cần kiểm tra:
1. Logic giá phải hợp lệ: High >= Open, Close, Low; Low <= Open, Close, High
2. Không có ngày trùng lặp
3. Giá/volume không được âm hoặc = 0 (lỗi dữ liệu từ nguồn)
4. Ngày phải liên tục theo lịch giao dịch (business days) - phát hiện khoảng
   trống bất thường (ngày nghỉ lễ dài, hoặc dữ liệu bị thiếu)
"""
import pandas as pd
import numpy as np


def load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def basic_data_quality_check(df: pd.DataFrame) -> dict:
    invalid_ohlc = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    ).sum()

    return {
        "n_rows": len(df),
        "date_range": (df["Date"].min(), df["Date"].max()),
        "n_duplicate_dates": df["Date"].duplicated().sum(),
        "n_invalid_ohlc": invalid_ohlc,
        "n_zero_or_negative": (df[["Open", "High", "Low", "Close", "Volume"]] <= 0).sum().sum(),
        "n_missing": df.isna().sum().sum(),
    }


def clean_price_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n_start = len(df)
    df = df.copy()

    # Bỏ ngày trùng lặp (giữ dòng đầu tiên)
    n_dup = df["Date"].duplicated().sum()
    df = df.drop_duplicates(subset=["Date"], keep="first")

    # Bỏ dòng có giá/volume <= 0 (lỗi dữ liệu)
    n_invalid_value = (df[["Open", "High", "Low", "Close", "Volume"]] <= 0).any(axis=1).sum()
    df = df[(df[["Open", "High", "Low", "Close", "Volume"]] > 0).all(axis=1)]

    # Bỏ dòng vi phạm logic OHLC (High phải là giá cao nhất, Low phải thấp nhất)
    invalid_ohlc = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )
    n_invalid_ohlc = invalid_ohlc.sum()
    df = df[~invalid_ohlc]

    df = df.sort_values("Date").reset_index(drop=True)

    if verbose:
        print(f"Số dòng ban đầu:              {n_start:,}")
        print(f"Bỏ do ngày trùng lặp:          {n_dup:,}")
        print(f"Bỏ do giá/volume <= 0:         {n_invalid_value:,}")
        print(f"Bỏ do vi phạm logic OHLC:      {n_invalid_ohlc:,}")
        print(f"Số dòng còn lại:               {len(df):,}")

    return df


def save_processed(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"Đã lưu -> {path}")


if __name__ == "__main__":
    raw = load_price_data("data/raw/stock_prices.csv")
    print("=== Data quality check ===")
    stats = basic_data_quality_check(raw)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print()
    clean = clean_price_data(raw)
    save_processed(clean, "data/processed/stock_prices_clean.csv")
