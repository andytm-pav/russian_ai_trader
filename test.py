#!/usr/bin/env python3
"""
Проверка: попадёт ли SBCB под фильтр exclude_sectors
"""

import json

# Загружаем ticker_sectors.json
with open("config/ticker_sectors.json", "r", encoding="utf-8") as f:
    sectors_data = json.load(f)

ticker_sectors = sectors_data.get("sectors", {})

# Загружаем settings.json (если уже обновлён)
with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

liquidity_filter = settings.get("liquidity_filter", {})
exclude_sectors = liquidity_filter.get("exclude_sectors", [])
allowed_boards = liquidity_filter.get("allowed_boards", [])

print("=" * 60)
print("ПРОВЕРКА ФИЛЬТРАЦИИ SBCB")
print("=" * 60)

# Информация о SBCB
sbcb_sector = ticker_sectors.get("SBCB", "НЕ НАЙДЕН")
print(f"\nТикер: SBCB")
print(f"Сектор в ticker_sectors.json: {sbcb_sector}")
print(f"Сектор в списке исключённых? {sbcb_sector in exclude_sectors}")

if sbcb_sector in exclude_sectors:
    print("\n✅ SBCB БУДЕТ ОТСЕЯН фильтром exclude_sectors")
else:
    print("\n❌ SBCB НЕ БУДЕТ ОТСЕЯН — сектор не в exclude_sectors")

# Статистика по исключаемым секторам
print(f"\nИсключаемые сектора: {exclude_sectors}")
print(f"Разрешённые boardid: {allowed_boards}")

if exclude_sectors:
    excluded_tickers = []
    for ticker, sector in ticker_sectors.items():
        if sector in exclude_sectors:
            excluded_tickers.append(ticker)

    print(f"\nТикеров, попадающих под исключение: {len(excluded_tickers)}")
    if excluded_tickers:
        print(f"Первые 30: {', '.join(sorted(excluded_tickers)[:30])}")