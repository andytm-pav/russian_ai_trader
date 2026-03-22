import requests
import pandas as pd
from datetime import datetime, timedelta


def get_brent_price():
    """Получение последней цены Brent с сайта EIA"""

    url = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"

    try:
        # Скачиваем Excel файл
        response = requests.get(url, timeout=15)

        # Читаем Excel, пропуская первые строки (там шапка)
        df = pd.read_excel(
            response.content,
            sheet_name=2,  # Лист с данными
            skiprows=4,  # Пропускаем заголовки
            names=['Date', 'Price']  # Называем колонки
        )

        # Убираем пустые строки
        df = df.dropna()

        # Последняя запись (самая свежая)
        latest = df.iloc[-1]
        date = latest['Date']
        price = latest['Price']

        print(f"✅ Brent: ${price} за баррель на {date}")

        return {
            'brent_usd': float(price),
            'brent_date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        }

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def get_brent_with_change():
    """Получение Brent и изменения за день"""

    url = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"

    try:
        df = pd.read_excel(requests.get(url).content, sheet_name=2, skiprows=4, names=['Date', 'Price'])
        df = df.dropna()

        # Последние две записи
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        current_price = float(latest['Price'])
        prev_price = float(previous['Price'])
        change = ((current_price - prev_price) / prev_price) * 100

        return {
            'brent_usd': current_price,
            'brent_prev': prev_price,
            'brent_change': change,
            'brent_date': latest['Date'].strftime('%Y-%m-%d')
        }

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Brent from EIA")
    print("=" * 50)

    data = get_brent_with_change()
    if data:
        print(f"\n📊 Brent: ${data['brent_usd']}")
        print(f"   Изменение: {data['brent_change']:+.2f}%")
        print(f"   Дата: {data['brent_date']}")