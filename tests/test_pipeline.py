"""Unit test cho pipeline xử lý dữ liệu, technical indicators, và backtest.
Chạy: pytest tests/ -v
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest

from src.data_processing import clean_price_data, basic_data_quality_check
from src.features import add_returns, add_moving_averages, add_volatility, add_rsi
from src.backtest import (
    generate_sma_crossover_signals, run_backtest, compute_performance_metrics
)
from src.forecasting import train_test_split_by_time, naive_baseline_forecast, evaluate_forecast


@pytest.fixture
def sample_price_df():
    dates = pd.bdate_range("2024-01-01", periods=100)
    np.random.seed(0)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 100)))
    return pd.DataFrame({
        "Date": dates,
        "Open": prices * 0.99,
        "High": prices * 1.02,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 5_000_000, 100),
    })


def test_clean_price_data_removes_duplicate_dates(sample_price_df):
    df_dup = pd.concat([sample_price_df, sample_price_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_price_data(df_dup, verbose=False)
    assert cleaned["Date"].duplicated().sum() == 0


def test_clean_price_data_removes_invalid_ohlc(sample_price_df):
    df = sample_price_df.copy()
    df.loc[5, "High"] = df.loc[5, "Low"] - 1  # High < Low -> lỗi logic rõ ràng
    cleaned = clean_price_data(df, verbose=False)
    assert len(cleaned) == len(df) - 1


def test_clean_price_data_removes_negative_values(sample_price_df):
    df = sample_price_df.copy()
    df.loc[10, "Volume"] = -100
    cleaned = clean_price_data(df, verbose=False)
    assert (cleaned["Volume"] > 0).all()


def test_data_quality_check_detects_duplicates(sample_price_df):
    df_dup = pd.concat([sample_price_df, sample_price_df.iloc[[0]]], ignore_index=True)
    stats = basic_data_quality_check(df_dup)
    assert stats["n_duplicate_dates"] == 1


def test_add_returns_first_value_is_nan(sample_price_df):
    result = add_returns(sample_price_df)
    assert pd.isna(result["DailyReturn"].iloc[0])
    assert result["DailyReturn"].iloc[1:].notna().all()


def test_add_returns_calculates_correctly():
    df = pd.DataFrame({"Close": [100, 110, 99]})
    result = add_returns(df)
    assert result["DailyReturn"].iloc[1] == pytest.approx(0.10)
    assert result["DailyReturn"].iloc[2] == pytest.approx(-0.10, abs=0.001)


def test_add_moving_averages_creates_correct_columns(sample_price_df):
    result = add_moving_averages(sample_price_df, windows=[10, 20])
    assert "SMA_10" in result.columns
    assert "SMA_20" in result.columns
    # SMA_20 tính được từ dòng thứ 20 trở đi (index 19)
    assert pd.isna(result["SMA_20"].iloc[18])
    assert result["SMA_20"].iloc[19] is not None and not pd.isna(result["SMA_20"].iloc[19])


def test_add_rsi_stays_within_bounds(sample_price_df):
    result = add_rsi(sample_price_df)
    valid_rsi = result["RSI"].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_add_volatility_is_non_negative(sample_price_df):
    df = add_returns(sample_price_df)
    result = add_volatility(df)
    valid_vol = result["Volatility"].dropna()
    assert (valid_vol >= 0).all()


def test_train_test_split_by_time_preserves_chronological_order(sample_price_df):
    train, test = train_test_split_by_time(sample_price_df, test_size=20)
    assert train["Date"].max() < test["Date"].min()  # train phải HOÀN TOÀN trước test
    assert len(test) == 20


def test_naive_baseline_forecast_returns_constant_last_price(sample_price_df):
    train = sample_price_df.iloc[:-10]
    forecast = naive_baseline_forecast(train["Close"], n_periods=10)
    assert len(forecast) == 10
    assert (forecast == train["Close"].iloc[-1]).all()


def test_evaluate_forecast_perfect_prediction_has_zero_error():
    y_true = np.array([100, 105, 98, 110])
    y_pred = np.array([100, 105, 98, 110])
    metrics = evaluate_forecast(y_true, y_pred, "Perfect Model")
    assert metrics["mae"] == pytest.approx(0)
    assert metrics["rmse"] == pytest.approx(0)


def test_sma_crossover_signal_shifted_to_avoid_lookahead_bias(sample_price_df):
    df = add_moving_averages(sample_price_df, windows=[10, 20])
    df = df.dropna(subset=["SMA_10", "SMA_20"]).reset_index(drop=True)
    result = generate_sma_crossover_signals(df, short_window=10, long_window=20)
    # Position tại dòng t phải bằng Signal tại dòng t-1 (đã shift), không phải Signal tại chính dòng t
    for i in range(1, len(result)):
        assert result["Position"].iloc[i] == result["Signal"].iloc[i - 1]


def test_run_backtest_deducts_transaction_cost(sample_price_df):
    df = add_returns(sample_price_df)
    df = add_moving_averages(df, windows=[10, 20])
    df = df.dropna(subset=["SMA_10", "SMA_20"]).reset_index(drop=True)
    df_signals = generate_sma_crossover_signals(df, short_window=10, long_window=20)
    result = run_backtest(df_signals, transaction_cost=0.01)
    # Ở những ngày có đổi vị thế, StrategyReturnNet phải NHỎ HƠN StrategyReturn (do trừ phí)
    changed = result[result["PositionChange"] > 0]
    if len(changed) > 0:
        assert (changed["StrategyReturnNet"] <= changed["StrategyReturn"]).all()


def test_compute_performance_metrics_max_drawdown_is_non_positive():
    df = pd.DataFrame({"StrategyReturnNet": [0.01, -0.02, 0.03, -0.05, 0.01]})
    metrics = compute_performance_metrics(df, return_col="StrategyReturnNet")
    assert metrics["max_drawdown"] <= 0  # drawdown luôn <= 0 theo định nghĩa


def test_compute_performance_metrics_win_rate_between_zero_and_one():
    df = pd.DataFrame({"StrategyReturnNet": [0.01, -0.02, 0.03, -0.05, 0.01, 0.02]})
    metrics = compute_performance_metrics(df, return_col="StrategyReturnNet")
    assert 0 <= metrics["win_rate"] <= 1
