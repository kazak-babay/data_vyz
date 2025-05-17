import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Загрузка файла
input_path = 'input.csv'
df = pd.read_csv(input_path)

# Преобразуем дату
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('D')

# === Разделим на train/test ===
# Обучаемся до 30 марта
train = df[:'2023-03-30']
# Тест — 31 марта + весь апрель
test = df['2023-03-31':'2023-04-30']

# === Прогноз на май (31 день) ===
forecast_horizon = 31
forecast_dates = pd.date_range(start='2023-05-01', periods=forecast_horizon, freq='D')
forecast_result = pd.DataFrame(index=forecast_dates, columns=df.columns)
metrics = []

for column in df.columns:
    series_train = train[column]
    series_test = test[column]

    model = ExponentialSmoothing(series_train, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(forecast_horizon)

    forecast_result[column] = forecast.values

    # Оцениваем качество модели на апреле (по test)
    test_forecast = model_fit.forecast(len(series_test))
    mae = mean_absolute_error(series_test, test_forecast)
    rmse = mean_squared_error(series_test, test_forecast) ** 0.5

    metrics.append({'column': column, 'MAE': round(mae, 6), 'RMSE': round(rmse, 6)})

# Округляем
forecast_result = forecast_result.round(6)
df = df.round(6)

# Объединяем
final_df = pd.concat([df, forecast_result])

# Сохраняем
final_output_path = "output.csv"
metrics_output_path = "metrics.csv"

final_df.to_csv(final_output_path)
pd.DataFrame(metrics).to_csv(metrics_output_path, index=False)