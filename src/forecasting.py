"""
Module dự báo giá cổ phiếu bằng 2 phương pháp time series kinh điển:
ARIMA và Prophet - so sánh hiệu năng bằng train/test split theo THỜI GIAN
(không phải random split như bài toán ML thông thường).

QUAN TRỌNG - vì sao không dùng random train/test split cho time series:
Dữ liệu time series có tính tự tương quan (autocorrelation) theo thời gian -
nếu chia ngẫu nhiên, model có thể "nhìn thấy" dữ liệu tương lai khi train
(data leakage nghiêm trọng). PHẢI train trên đoạn thời gian sớm hơn, test
trên đoạn thời gian sau đó - mô phỏng đúng tình huống dự báo thực tế (chỉ
biết quá khứ, không biết tương lai).

ARIMA (AutoRegressive Integrated Moving Average):
- Giả định dữ liệu (sau khi lấy sai phân - "Integrated") có tính dừng
  (stationary) - trung bình và phương sai không đổi theo thời gian
- Phù hợp với chuỗi có pattern tuyến tính rõ ràng, ít yếu tố mùa vụ phức tạp

Prophet (do Meta phát triển):
- Xử lý tốt seasonality (mùa vụ) và ngày lễ, ít cần tinh chỉnh tham số
- Phù hợp với dữ liệu kinh doanh có chu kỳ rõ ràng (ít phù hợp hơn với giá cổ
  phiếu thuần túy vì giá cổ phiếu gần với random walk, ít mùa vụ - nhưng vẫn
  đưa vào so sánh để minh họa sự khác biệt giữa 2 phương pháp)
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


def train_test_split_by_time(df: pd.DataFrame, test_size: int = 30) -> tuple:
    """Chia theo THỜI GIAN: test_size ngày CUỐI CÙNG làm tập test, phần còn
    lại (sớm hơn) làm tập train - không được xáo trộn ngẫu nhiên."""
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def fit_arima(train_prices: pd.Series, order: tuple = (5, 1, 0)):
    """order=(p,d,q): p=số lag tự hồi quy, d=bậc sai phân để đạt stationary,
    q=số lag của moving average. (5,1,0) là lựa chọn khởi điểm hợp lý cho giá
    cổ phiếu: d=1 vì giá thường cần lấy sai phân 1 lần mới stationary."""
    model = ARIMA(train_prices, order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(fitted_model, n_periods: int) -> pd.Series:
    forecast = fitted_model.forecast(steps=n_periods)
    return forecast


def fit_prophet(train_df: pd.DataFrame):
    """Prophet yêu cầu đúng format cột: 'ds' (ngày) và 'y' (giá trị dự báo)."""
    from prophet import Prophet

    prophet_df = train_df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    return model


def forecast_prophet(model, n_periods: int, freq: str = "B") -> pd.DataFrame:
    """freq='B' = business day (ngày làm việc), khớp với lịch giao dịch chứng khoán."""
    future = model.make_future_dataframe(periods=n_periods, freq=freq)
    forecast = model.predict(future)
    return forecast.tail(n_periods)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))) * 100

    print(f"=== {model_name} ===")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {"model_name": model_name, "mae": mae, "rmse": rmse, "mape": mape}


def naive_baseline_forecast(train_prices: pd.Series, n_periods: int) -> np.ndarray:
    """Baseline đơn giản nhất: dự đoán giá ngày mai = giá hôm nay (random walk
    naive forecast). Model ARIMA/Prophet PHẢI đánh bại được baseline này mới
    có giá trị thực tế - đây là bước kiểm tra sanity check quan trọng thường
    bị bỏ qua khi làm forecasting."""
    last_price = train_prices.iloc[-1]
    return np.full(n_periods, last_price)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/stock_prices_clean.csv", parse_dates=["Date"])
    train, test = train_test_split_by_time(df, test_size=30)

    print(f"Train: {len(train)} ngày ({train['Date'].min().date()} -> {train['Date'].max().date()})")
    print(f"Test:  {len(test)} ngày ({test['Date'].min().date()} -> {test['Date'].max().date()})\n")

    # Baseline
    naive_pred = naive_baseline_forecast(train["Close"], len(test))
    naive_metrics = evaluate_forecast(test["Close"].values, naive_pred, "Naive Baseline (giá không đổi)")

    # ARIMA
    arima_model = fit_arima(train["Close"])
    arima_pred = forecast_arima(arima_model, len(test))
    arima_metrics = evaluate_forecast(test["Close"].values, arima_pred.values, "ARIMA(5,1,0)")

    # Prophet
    prophet_model = fit_prophet(train)
    prophet_forecast = forecast_prophet(prophet_model, len(test))
    prophet_metrics = evaluate_forecast(test["Close"].values, prophet_forecast["yhat"].values, "Prophet")

    print("\n=== Tóm tắt so sánh ===")
    summary = pd.DataFrame([naive_metrics, arima_metrics, prophet_metrics])
    print(summary.to_string(index=False))
    summary.to_csv("outputs/forecast_comparison.csv", index=False)
