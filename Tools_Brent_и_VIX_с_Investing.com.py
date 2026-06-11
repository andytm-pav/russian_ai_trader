"""
Получение Brent и VIX с Investing.com
"""
import requests
import re
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_brent():
    """Парсинг цены Brent с Investing.com"""
    url = "https://ru.investing.com/commodities/brent-oil"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"Brent: HTTP {resp.status_code}")
            return None

        # Ищем цену через data-test атрибут
        match = re.search(r'data-test="instrument-price-last">\s*([\d.,]+)', resp.text)
        if match:
            raw = match.group(1).replace('.', '').replace(',', '.')
            return float(raw)

        # Запасной вариант — ищем в JSON-блоке
        match = re.search(r'"last":([\d.]+)', resp.text)
        if match:
            return float(match.group(1))

        print("Brent: цена не найдена в HTML")
        return None

    except Exception as e:
        print(f"Brent: ошибка — {e}")
        return None


def get_vix():
    """Парсинг VIX с Investing.com"""
    url = "https://ru.investing.com/indices/volatility-s-p-500"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"VIX: HTTP {resp.status_code}")
            return None

        match = re.search(r'data-test="instrument-price-last">\s*([\d.,]+)', resp.text)
        if match:
            raw = match.group(1).replace('.', '').replace(',', '.')
            return float(raw)

        match = re.search(r'"last":([\d.]+)', resp.text)
        if match:
            return float(match.group(1))

        print("VIX: цена не найдена в HTML")
        return None

    except Exception as e:
        print(f"VIX: ошибка — {e}")
        return None


# Тест
print("=" * 50)
print("ТЕСТ INVESTING.COM")
print("=" * 50)

for i in range(3):
    print(f"\nЗапрос #{i+1}:")
    start = time.time()

    brent = get_brent()
    if brent:
        print(f"  Brent: ${brent:.2f} (за {time.time() - start:.1f}с)")
    else:
        print(f"  Brent: ❌ (за {time.time() - start:.1f}с)")

    start = time.time()
    vix = get_vix()
    if vix:
        print(f"  VIX:   {vix:.2f} (за {time.time() - start:.1f}с)")
    else:
        print(f"  VIX:   ❌ (за {time.time() - start:.1f}с)")

    if i < 2:
        time.sleep(2)

print("\nГотово.")