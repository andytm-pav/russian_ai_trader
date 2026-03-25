# diagnostic_liquidity.py
"""
Проверка доступности данных для признаков:
1. Оборот рынка акций (shares_turnover)
2. Рыночная капитализация (market_cap)
3. Средний оборот за 5 дней (avg_turnover_5d)
"""

import requests
import json
from datetime import datetime, timedelta


def check_shares_turnover():
    """Проверка оборота рынка акций"""
    print("\n" + "=" * 60)
    print("📊 ОБОРОТ РЫНКА АКЦИЙ")
    print("=" * 60)

    url = "https://iss.moex.com/iss/engines/stock/turnovers.json"
    params = {'iss.meta': 'off', 'iss.only': 'turnovers'}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'turnovers' in data and data['turnovers']['data']:
            cols = data['turnovers']['columns']
            print(f"📋 Колонки: {cols}")

            for row in data['turnovers']['data']:
                name = row[cols.index('NAME')]
                if name == 'shares':
                    value = row[cols.index('VALTODAY')]
                    print(f"✅ Оборот акций: {value:,.0f} ₽")
                    return value
            print("❌ Рынок акций не найден")
        else:
            print("❌ Нет данных оборотов")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    return None


def check_market_capitalization():
    """Проверка рыночной капитализации"""
    print("\n" + "=" * 60)
    print("📊 РЫНОЧНАЯ КАПИТАЛИЗАЦИЯ")
    print("=" * 60)

    url = "https://iss.moex.com/iss/statistics/engines/stock/capitalization.json"
    params = {'iss.meta': 'off', 'iss.only': 'capitalization'}

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'capitalization' in data and data['capitalization']['data']:
            cols = data['capitalization']['columns']
            print(f"📋 Колонки: {cols}")

            for row in data['capitalization']['data']:
                print(f"   {row}")

            # Берем первую строку
            row = data['capitalization']['data'][0]
            if 'value' in cols:
                idx = cols.index('value')
                cap = row[idx]
                print(f"✅ Капитализация: {cap:,.0f} ₽")
                return cap
        else:
            print("❌ Нет данных капитализации")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    return None


def check_historical_turnover(days: int = 5):
    """Проверка исторических оборотов (через исторические данные)"""
    print("\n" + "=" * 60)
    print(f"📊 ИСТОРИЧЕСКИЕ ОБОРОТЫ (последние {days} дней)")
    print("=" * 60)

    # Эндпоинт исторических данных (требует подписки)
    url = "https://iss.moex.com/iss/history/engines/stock/turnovers.json"
    params = {
        'iss.meta': 'off',
        'iss.only': 'turnovers',
        'limit': days
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'turnovers' in data and data['turnovers']['data']:
            cols = data['turnovers']['columns']
            print(f"📋 Колонки: {cols}")

            turnovers = []
            for row in data['turnovers']['data']:
                # Ищем shares
                if len(row) > 0 and row[0] == 'shares':
                    if 'VALTODAY' in cols:
                        idx = cols.index('VALTODAY')
                        turnovers.append(row[idx])

            if turnovers:
                print(f"✅ Найдено оборотов: {len(turnovers)}")
                for i, t in enumerate(turnovers):
                    print(f"   День {i + 1}: {t:,.0f} ₽")
                avg = sum(turnovers) / len(turnovers)
                print(f"📊 Средний за {len(turnovers)} дней: {avg:,.0f} ₽")
                return avg
            else:
                print("❌ Данные по акциям не найдены")
        else:
            print("❌ Нет исторических данных (возможно, требуется подписка)")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    return None


def main():
    print("\n" + "█" * 60)
    print("  ДИАГНОСТИКА ДАННЫХ ДЛЯ ПРИЗНАКОВ ЛИКВИДНОСТИ")
    print(f"  Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 60)

    turnover = check_shares_turnover()
    cap = check_market_capitalization()
    avg_turnover = check_historical_turnover(5)

    print("\n" + "=" * 60)
    print("ИТОГ:")
    if turnover and cap:
        liquidity_ratio = turnover / cap if cap > 0 else 0
        print(f"✅ Коэффициент ликвидности (оборот/кап): {liquidity_ratio:.6f}")
    else:
        print("❌ Недостаточно данных для расчета коэффициента ликвидности")

    if turnover and avg_turnover:
        turnover_ratio = turnover / avg_turnover if avg_turnover > 0 else 0
        print(f"✅ Относительный оборот (сегодня/средний): {turnover_ratio:.2f}")
    else:
        print("❌ Недостаточно данных для расчета относительного оборота")
    print("=" * 60)


if __name__ == "__main__":
    main()