from fetchers.moex_fetcher import MoexFetcher
import pandas as pd

moex = MoexFetcher()

# Проверка 1: Базовая цена
print("=== ПРОВЕРКА GAZP ===")
price = moex.get_price("GAZP")
print(f"Текущая цена GAZP: {price}")

# Проверка 2: Свечи
candles = moex.get_candles("GAZP", interval=60, count=10)
print(f"Свечи GAZP: {'Есть' if candles is not None and not candles.empty else 'Нет'}")
if candles is not None:
    print(f"Размер: {candles.shape}")
    print(f"Колонки: {list(candles.columns)}")
    print(f"Первые строки:\n{candles.head()}")

# Проверка 3: Информация
info = moex.get_ticker_info("GAZP")
print(f"Информация GAZP: {'Есть' if info else 'Нет'}")