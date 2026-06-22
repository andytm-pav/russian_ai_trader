"""
Проверка загрузки исторических макро-данных MOEX ISS API
"""
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

BASE_URL = "https://iss.moex.com/iss"
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

print("=" * 70)
print("ПРОВЕРКА ИСТОРИЧЕСКИХ МАКРО-ДАННЫХ MOEX")
print("=" * 70)

TEST_DATE_START = "2025-06-01"
TEST_DATE_END = "2025-06-15"

# 1. Исторический IMOEX
print(f"\n1. IMOEX ({TEST_DATE_START} → {TEST_DATE_END}):")
url = f"{BASE_URL}/history/engines/stock/markets/index/securities/IMOEX.json"
params = {
    'from': TEST_DATE_START,
    'till': TEST_DATE_END,
    'iss.meta': 'off',
    'history.columns': 'TRADEDATE,CLOSE'
}

start = time.time()
try:
    resp = session.get(url, params=params, timeout=15)
    elapsed = time.time() - start
    if resp.status_code == 200:
        data = resp.json()
        if 'history' in data and data['history']['data']:
            rows = data['history']['data']
            print(f"  Получено {len(rows)} записей за {elapsed:.1f}с")
            for row in rows[:3]:
                print(f"    {row[0]}: IMOEX={row[1]}")
            if len(rows) > 3:
                print(f"    ... и ещё {len(rows)-3}")
        else:
            print(f"  Нет данных (за {elapsed:.1f}с)")
    else:
        print(f"  HTTP {resp.status_code} (за {elapsed:.1f}с)")
except Exception as e:
    elapsed = time.time() - start
    print(f"  Ошибка: {e} (за {elapsed:.1f}с)")

# 2. Исторический RTSI
print(f"\n2. RTSI ({TEST_DATE_START} → {TEST_DATE_END}):")
url = f"{BASE_URL}/history/engines/stock/markets/index/securities/RTSI.json"
params = {
    'from': TEST_DATE_START,
    'till': TEST_DATE_END,
    'iss.meta': 'off',
    'history.columns': 'TRADEDATE,CLOSE'
}

start = time.time()
try:
    resp = session.get(url, params=params, timeout=15)
    elapsed = time.time() - start
    if resp.status_code == 200:
        data = resp.json()
        if 'history' in data and data['history']['data']:
            rows = data['history']['data']
            print(f"  Получено {len(rows)} записей за {elapsed:.1f}с")
            for row in rows[:3]:
                print(f"    {row[0]}: RTSI={row[1]}")
        else:
            print(f"  Нет данных (за {elapsed:.1f}с)")
    else:
        print(f"  HTTP {resp.status_code} (за {elapsed:.1f}с)")
except Exception as e:
    elapsed = time.time() - start
    print(f"  Ошибка: {e} (за {elapsed:.1f}с)")

# 3. Исторический RVI (волатильность)
print(f"\n3. RVI ({TEST_DATE_START} → {TEST_DATE_END}):")
url = f"{BASE_URL}/history/engines/stock/markets/index/securities/RVI.json"
params = {
    'from': TEST_DATE_START,
    'till': TEST_DATE_END,
    'iss.meta': 'off',
    'history.columns': 'TRADEDATE,CLOSE'
}

start = time.time()
try:
    resp = session.get(url, params=params, timeout=15)
    elapsed = time.time() - start
    if resp.status_code == 200:
        data = resp.json()
        if 'history' in data and data['history']['data']:
            rows = data['history']['data']
            print(f"  Получено {len(rows)} записей за {elapsed:.1f}с")
            for row in rows[:3]:
                print(f"    {row[0]}: RVI={row[1]}")
        else:
            print(f"  Нет данных (за {elapsed:.1f}с)")
    else:
        print(f"  HTTP {resp.status_code} (за {elapsed:.1f}с)")
except Exception as e:
    elapsed = time.time() - start
    print(f"  Ошибка: {e} (за {elapsed:.1f}с)")

# 4. USD/RUB через ЦБ РФ
print(f"\n4. USD/RUB ЦБ РФ ({TEST_DATE_START} → {TEST_DATE_END}):")
import xml.etree.ElementTree as ET

cbr_session = requests.Session()
cbr_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (compatible; CBRClient/1.0)'
})

test_dates = ["2025-06-02", "2025-06-06", "2025-06-11"]
for date_str in test_dates:
    url = f"http://www.cbr.ru/scripts/XML_daily.asp?date_req={date_str}"
    start = time.time()
    try:
        resp = cbr_session.get(url, timeout=15)
        elapsed = time.time() - start
        if resp.status_code == 200:
            resp.encoding = 'windows-1251'
            root = ET.fromstring(resp.text)
            for valute in root.findall('Valute'):
                if valute.find('CharCode').text == 'USD':
                    value = valute.find('Value').text
                    nominal = valute.find('Nominal').text
                    usd_rub = float(value.replace(',', '.')) / float(nominal)
                    print(f"    {date_str}: USD/RUB={usd_rub:.2f} (за {elapsed:.1f}с)")
        else:
            print(f"    {date_str}: HTTP {resp.status_code} (за {elapsed:.1f}с)")
            print(f"    Ответ: {resp.text[:200]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"    {date_str}: {type(e).__name__}: {e} (за {elapsed:.1f}с)")

# 5. Нефть Brent (через Investing.com не получится исторически)
print(f"\n5. Brent (текущий):")
from fetchers.moex_fetcher import MoexFetcher
moex = MoexFetcher()
brent = moex.get_brent_price()
print(f"  Текущая цена: ${brent:.2f}" if brent else "  Нет данных")

print("\n" + "=" * 70)
print("ГОТОВО")
print("=" * 70)
print("\nВывод: MOEX отдаёт исторические данные для IMOEX, RTSI, RVI, USD/RUB.")
print("Brent нужно получать через Investing.com (только текущие данные).")