#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ТЕСТОВЫЙ СКРИПТ для получения данных:
- USD/RUB и ключевая ставка (ЦБ РФ)
- Brent и цены на нефть (OilPriceAPI)
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json


def test_cbr_data():
    """Тест получения данных с сайта ЦБ РФ"""

    print("\n" + "=" * 60)
    print("ТЕСТ 1: ПОЛУЧЕНИЕ ДАННЫХ С ЦБ РФ")
    print("=" * 60)

    results = {}

    # 1.1 Курс USD/RUB
    print("\n📈 1.1 Получение USD/RUB:")
    url = "http://www.cbr.ru/scripts/XML_daily.asp"

    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'windows-1251'
        root = ET.fromstring(response.text)

        for valute in root.findall('Valute'):
            char_code = valute.find('CharCode').text
            if char_code == 'USD':
                value = valute.find('Value').text
                nominal = valute.find('Nominal').text
                # Заменяем запятую на точку и конвертируем
                usd_rub = float(value.replace(',', '.')) / float(nominal)
                results['usd_rub'] = usd_rub
                print(f"   ✅ USD/RUB: {usd_rub:.4f}")
                break
        else:
            print("   ❌ USD не найден")

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    # 1.2 Ключевая ставка (через XML_depo)
    print("\n📊 1.2 Получение ключевой ставки:")

    # Запрашиваем за последние 5 дней
    date_from = (datetime.now() - timedelta(days=5)).strftime('%d/%m/%Y')
    date_to = datetime.now().strftime('%d/%m/%Y')

    url = "http://www.cbr.ru/scripts/XML_depo.asp"
    params = {
        'date_req1': date_from,
        'date_req2': date_to
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.encoding = 'windows-1251'
        root = ET.fromstring(response.text)

        # Берем самую свежую запись
        latest = root.find('Record')
        if latest is not None:
            overnight = latest.find('Overnight')
            if overnight is not None:
                rate = float(overnight.text.replace(',', '.'))
                results['cbr_rate'] = rate
                print(f"   ✅ Ключевая ставка (Overnight): {rate}%")
            else:
                print("   ❌ Нет данных Overnight")
        else:
            print("   ❌ Нет записей")

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

    return results


def test_oil_data():
    """Тест получения цен на нефть"""

    print("\n" + "=" * 60)
    print("ТЕСТ 2: ПОЛУЧЕНИЕ ЦЕН НА НЕФТЬ")
    print("=" * 60)

    results = {}

    # 2.1 Бесплатный API (данные с задержкой)
    print("\n🛢️ 2.1 Альтернативный источник (бесплатный):")

    # Пробуем разные бесплатные API
    apis = [
        {
            'name': 'EIA',
            'url': 'https://api.eia.gov/v2/petroleum/pri/fut/data/',
            'params': {
                'frequency': 'daily',
                'data[0]': 'value',
                'facets[product][]': 'RBRTE',  # Brent
                'start': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'sort[0][column]': 'period',
                'sort[0][direction]': 'desc',
                'offset': 0,
                'length': 1
            }
        },
        {
            'name': 'RapidAPI (OilPriceAPI)',
            'url': 'https://oil-price-api.p.rapidapi.com/oil-price',
            'headers': {
                'X-RapidAPI-Key': 'demo_key',  # Нужен реальный ключ
                'X-RapidAPI-Host': 'oil-price-api.p.rapidapi.com'
            }
        }
    ]

    for api in apis:
        print(f"\n   📡 Тест: {api['name']}")
        try:
            if 'headers' in api:
                response = requests.get(api['url'], headers=api['headers'], timeout=10)
            else:
                response = requests.get(api['url'], params=api.get('params', {}), timeout=10)

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Статус 200")
                print(f"   📄 Ответ: {json.dumps(data, indent=2)[:200]}...")

                # Здесь нужно будет парсить под конкретный API
                if 'brent' in data:
                    results['brent'] = data['brent']

            else:
                print(f"   ❌ Статус {response.status_code}")

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    # 2.2 Показываем, какие ключи нужны для OilPriceAPI
    print("\n🔑 Для работы с OilPriceAPI нужен ключ:")
    print("   1. Зарегистрироваться на https://rapidapi.com/")
    print("   2. Подписаться на OilPrice API (есть бесплатный tier)")
    print("   3. Использовать ключ в заголовках:")
    print("      headers = {")
    print("          'X-RapidAPI-Key': 'ваш_ключ',")
    print("          'X-RapidAPI-Host': 'oil-price-api.p.rapidapi.com'")
    print("      }")

    return results


def test_macro_data():
    """Сбор всех макро-данных"""

    print("\n" + "=" * 60)
    print("СБОР ВСЕХ МАКРО-ДАННЫХ")
    print("=" * 60)

    macro_data = {
        # MOEX данные (уже есть)
        'imoex': 2851.37,
        'imoex_change': -0.71,
        'rtsi': 1109.15,
        'rtsi_change': -1.64,
        'rvi': 24.49,
        'rvi_change': 2.43,

        # ЦБ РФ данные
        'usd_rub': 0,
        'usd_rub_change': 0,
        'cbr_rate': 0,

        # Нефть
        'brent': 0,
        'brent_change': 0,
        'oil_price': 0,

        # Прочее
        'inflation': 0,
        'market_volatility': 0
    }

    # Получаем данные ЦБ
    cbr = test_cbr_data()
    if 'usd_rub' in cbr:
        macro_data['usd_rub'] = cbr['usd_rub']
    if 'cbr_rate' in cbr:
        macro_data['cbr_rate'] = cbr['cbr_rate']

    # Получаем данные по нефти
    oil = test_oil_data()
    if 'brent' in oil:
        macro_data['brent'] = oil['brent']

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МАКРО-ДАННЫЕ:")
    for key, value in macro_data.items():
        if value != 0:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value} (нет данных)")

    return macro_data


if __name__ == "__main__":
    print("🚀 ЗАПУСК ТЕСТОВ МАКРО-ДАННЫХ")
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    macro = test_macro_data()

    print("\n" + "=" * 60)
    print("✅ ТЕСТЫ ЗАВЕРШЕНЫ")