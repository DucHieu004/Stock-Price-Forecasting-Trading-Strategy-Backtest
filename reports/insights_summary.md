# Executive Summary — Stock Forecasting & Strategy Backtest

> Báo cáo tóm tắt kết quả nghiên cứu, dựa trên kết quả chạy pipeline (xem chi
> tiết kỹ thuật tại `notebooks/`).
> **Số liệu dưới đây từ bộ dữ liệu MẪU (synthetic)** — cần chạy lại report
> này sau khi thay bằng dữ liệu thị trường thật (qua `yfinance`).

## 1. Câu hỏi nghiên cứu

1. Có thể dự báo giá cổ phiếu tốt hơn phương pháp "đoán bừa" đơn giản nhất không?
2. Chiến lược giao dịch dựa trên tín hiệu kỹ thuật có tạo ra giá trị thực sự
   so với chỉ mua và giữ (Buy & Hold) không?

## 2. Kết quả Forecasting

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive Baseline (giá không đổi) | 6.11 | 7.72 | 2.92% |
| **ARIMA(5,1,0)** | **5.02** | **6.33** | **2.39%** |
| Prophet | 11.35 | 13.54 | 5.29% |

**Insight chính**: ARIMA giảm RMSE khoảng **18%** so với baseline — bằng
chứng có autocorrelation ngắn hạn trong giá mà mô hình thống kê đơn giản có
thể khai thác được.

**Phát hiện đáng chú ý**: Prophet — công cụ forecasting phổ biến do Meta phát
triển, tối ưu cho dữ liệu có seasonality/ngày lễ rõ ràng (VD: doanh số bán
hàng, lượt truy cập web) — hoạt động **kém hơn cả baseline đơn giản nhất**
trên dữ liệu giá cổ phiếu. Nguyên nhân: giá cổ phiếu gần với quá trình random
walk, thiếu pattern mùa vụ rõ ràng để Prophet khai thác lợi thế của nó.

**Bài học phương pháp luận**: không có công cụ nào "tốt nhất" cho mọi loại dữ
liệu — phải hiểu đặc điểm thống kê của dữ liệu (đã kiểm định bằng ADF test ở
bước Technical Analysis) trước khi chọn phương pháp mô hình hóa phù hợp.

## 3. Kết quả Backtest chiến lược

| Chiến lược | Total Return | Annualized Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| **SMA Crossover (20/50)** | **+122.0%** | **+33.2%** | **1.08** | **-36.1%** |
| Buy & Hold | +71.1% | +21.3% | 0.45 | -45.7% |

**Insight chính**: chiến lược SMA Crossover vượt trội Buy & Hold trên cả 3
khía cạnh — lợi nhuận cao hơn, Sharpe Ratio gấp 2.4 lần (lợi nhuận điều chỉnh
theo rủi ro tốt hơn nhiều), và Max Drawdown thấp hơn (chịu ít rủi ro "sập"
danh mục hơn trong giai đoạn thị trường xấu).

**Cơ chế tạo ra lợi thế**: chiến lược tự động THOÁT vị thế khi xu hướng đảo
chiều (SMA ngắn cắt xuống dưới SMA dài — "Death Cross"), giúp tránh phần lớn
mức sụt giảm trong giai đoạn thị trường xấu, đổi lại bằng việc bỏ lỡ một phần
lợi nhuận khi thị trường phục hồi nhanh (vào lại vị thế trễ hơn Buy & Hold).

## 4. Kiểm tra độ tin cậy: Parameter Sensitivity

Đã backtest chiến lược với nhiều cặp tham số SMA khác nhau (short window:
10/20/30, long window: 50/80/100) thay vì chỉ báo cáo 1 cặp tham số duy nhất.
Kết quả Sharpe Ratio ổn định qua các cặp tham số khác nhau (xem chi tiết
`outputs/parameter_sensitivity.csv`) — giảm rủi ro kết luận dựa trên 1 cặp
tham số "may mắn" (overfitting vào dữ liệu lịch sử).

## 5. Giới hạn quan trọng (không phóng đại kết quả)

1. **Dữ liệu mẫu, không phải thị trường thật** — kết quả cần xác nhận lại
   trên dữ liệu thật trước khi có ý nghĩa cho quyết định đầu tư
2. **Không có slippage** — giả định khớp lệnh đúng giá đóng cửa, thực tế có
   thể lệch, đặc biệt với khối lượng giao dịch lớn
3. **Chỉ test 1 giai đoạn thị trường** — cần test trên nhiều giai đoạn
   (bull/bear/sideways market) để đánh giá tính ổn định của chiến lược
4. **Chưa tính thuế, phí lưu ký, lãi vay margin** (nếu dùng đòn bẩy)

## 6. Định hướng nghiên cứu tiếp theo

1. **Walk-forward backtesting**: huấn luyện lại model định kỳ theo thời gian
   thay vì chỉ train 1 lần duy nhất — mô phỏng sát cách vận hành thực tế hơn
2. **Đa dạng hóa universe test**: chạy trên nhiều mã cổ phiếu, nhiều giai
   đoạn thị trường khác nhau để đánh giá tính ổn định (robustness)
3. **Kết hợp thêm tín hiệu**: MACD, Bollinger Bands, hoặc ML-based signal
   (feature từ technical indicators + model classification) để so sánh với
   chiến lược rule-based đơn giản hiện tại

---
*Báo cáo này sinh ra từ `src/forecasting.py`, `src/backtest.py`,
`notebooks/05_conclusion.ipynb`. Chạy lại bằng dữ liệu thị trường thật trước
khi dùng cho bất kỳ quyết định đầu tư nào.*
