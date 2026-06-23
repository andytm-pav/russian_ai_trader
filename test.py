#!/usr/bin/env python3
"""
Проверка: есть ли BOARDID в ответе MOEX ISS при запросе списка бумаг
"""

import requests

URL = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
PARAMS = {
    'iss.meta': 'off',
    'iss.only': 'securities',
    'securities.columns': 'SECID,SHORTNAME,BOARDID',
    'limit': 10
}

try:
    resp = requests.get(URL, params=PARAMS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
except Exception as e:
    print(f"Ошибка запроса: {e}")
    exit()

cols = data['securities']['columns']
rows = data['securities']['data']

print(f"Колонки: {cols}")
print(f"Первые 5 строк:")
for row in rows[:5]:
    print(f"  {row}")

if 'BOARDID' in cols:
    print("\n✅ BOARDID присутствует в ответе API")
else:
    print("\n❌ BOARDID отсутствует в ответе API")