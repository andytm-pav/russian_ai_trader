#!/usr/bin/env python3
"""
Тест: запрос карточек конкретных бумаг для поиска типа инструмента
"""

import requests

TICKERS = [
    "SBER",   # акция обыкновенная
    "SBERP",  # акция привилегированная
    "SBMX",   # БПИФ фонд
    "CASH",   # ETF фонд
    "RU000A108ZB2",  # облигация (если есть)
]

URL_TEMPLATE = "https://iss.moex.com/iss/securities/{ticker}.json"

print("Проверка полей в карточках бумаг...\n")

for ticker in TICKERS:
    url = URL_TEMPLATE.format(ticker=ticker)
    try:
        resp = requests.get(url, params={'iss.meta': 'off'}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"{ticker}: ошибка запроса - {e}")
        continue

    # Ищем секцию description или boards
    description = data.get('description', {})
    boards = data.get('boards', {})

    print(f"\n{'='*60}")
    print(f"Тикер: {ticker}")

    # Поля из description
    desc_data = description.get('data', [])
    desc_cols = description.get('columns', [])
    if desc_data and desc_cols:
        row = desc_data[0]
        print("  Поля description:")
        for col, val in zip(desc_cols, row):
            if val is not None:
                print(f"    {col}: {val}")

    # Поля из boards
    boards_data = boards.get('data', [])
    boards_cols = boards.get('columns', [])
    if boards_data and boards_cols:
        row = boards_data[0]
        print("  Поля boards:")
        for col, val in zip(boards_cols, row):
            if val is not None:
                print(f"    {col}: {val}")