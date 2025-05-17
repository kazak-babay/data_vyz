import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# === 1. Загрузка данных ===
df = pd.read_csv('public_data.csv')

# Преобразуем столбец 'date' в формат datetime и делаем его индексом
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Задаём дневную частоту (важно для прогнозов)
df = df.asfreq('D')

# === 2. Определим обучающую выборку ===
train = df[df.index < '2023-05-01']

# Горизонт прогноза — 31 день (на май 2023)
forecast_horizon = 31
forecast_dates = pd.date_range(start='2023-05-01', periods=forecast_horizon, freq='D')

# Пустой датафрейм для прогнозов
forecast_result = pd.DataFrame(index=forecast_dates, columns=df.columns)

# === 3. Прогнозирование каждого временного ряда ===
for column in df.columns:
    series_train = train[column]

    # Простая модель с аддитивным трендом
    model = ExponentialSmoothing(series_train, trend='add', seasonal=None)
    model_fit = model.fit()

    forecast = model_fit.forecast(forecast_horizon)
    forecast_result[column] = forecast.values

# === 4. Объединяем исходные данные и прогноз ===
final_df = pd.concat([df, forecast_result])

# === 5. Сохраняем результат ===
final_df = final_df.round(6)
final_df.to_csv('output.csv')

print("✅ Прогноз сохранён в файл output.csv")
