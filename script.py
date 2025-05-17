# Перезапуск окружения сбросил все данные — нужно повторно загрузить и обработать файл
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Загрузка файла
input_path = 'public_data.csv'
df = pd.read_csv(input_path)

# Преобразуем дату
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('D')

# Разделим на train/test
test_start = '2023-04-01'
test_end = '2023-04-30'
test = df[test_start:test_end]
train = df[:'2023-03-30']

# Прогнозируем на апрель (чтобы сравнить с тестом)
forecast_horizon = 30
forecast_dates = pd.date_range(start='2023-04-01', periods=forecast_horizon, freq='D')

forecast_result = pd.DataFrame(index=forecast_dates, columns=df.columns)
metrics = []

for column in df.columns:
    series_train = train[column]
    series_test = test[column]

    model = ExponentialSmoothing(series_train, trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(forecast_horizon)

    forecast_result[column] = forecast.values

    mae = mean_absolute_error(series_test, forecast)
    mse = mean_squared_error(series_test, forecast)
    rmse = mse ** 0.5

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

final_output_path, metrics_output_path