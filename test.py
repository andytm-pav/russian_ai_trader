# verify_brent_price.py
"""
Верификация цены Brent через альтернативные источники
"""

import requests
import re
from datetime import datetime


def check_finmarket_alternative():
    """Проверка других разделов Finmarket"""
    urls = [
        "https://www.finmarket.ru/",
        "https://www.finmarket.ru/commodities/",
        "https://www.finmarket.ru/news/",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            text = resp.text

            # Поиск с контекстом "Brent"
            matches = re.findall(r'Brent[^<]*([\d,]+\.?\d*)[^<]*\$', text, re.IGNORECASE)
            if matches:
                price = matches[0].replace(',', '.')
                print(f"🔍 {url}: Brent = {price} USD")
                return float(price)
        except:
            continue
    return None


def check_moex_index():
    """Проверка MOEXOG как индикатора"""
    url = "https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/MOEXOG.json"
    params = {'iss.meta': 'off', 'iss.only': 'marketdata'}

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if 'marketdata' in data and data['marketdata']['data']:
            value = data['marketdata']['data'][0][1]
            print(f"📊 MOEXOG индекс: {value} (прокси для Brent)")
            return value
    except:
        pass
    return None


def manual_expected_price():
    """Ожидаемая цена из открытых источников"""
    # По данным на 26.03.2026: Brent ~$102-119
    print("📰 По данным новостей (26.03.2026):")
    print("   - Finmarket: Brent превысил $104,1")
    print("   - TradingView: диапазон $96–119")
    print("   - Ожидаемая цена: ~$102-110")
    return 105.0


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  ВЕРИФИКАЦИЯ ЦЕНЫ BRENT")
    print(f"  Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 60)

    price = check_finmarket_alternative()
    if price:
        print(f"\n✅ Найдена цена: {price} USD")
    else:
        print("\n❌ Точная цена не найдена")

    moexog = check_moex_index()
    if moexog:
        print(f"📊 MOEXOG: {moexog}")

    expected = manual_expected_price()
    print(f"\n🎯 Ожидаемый диапазон: $102-110")

    print("\n" + "=" * 60)
    print("ВЫВОД: Необходимо найти источник с Brent, а не WTI")
    print("=" * 60)