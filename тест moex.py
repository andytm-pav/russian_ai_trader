import requests
import json


def test_moex_api():
    """Простой тест доступности MOEX API"""

    print("Тестирование базового доступа к MOEX API...")

    # Тест 1: Проверяем общую доступность
    try:
        url = "https://iss.moex.com/iss/engines.json"
        response = requests.get(url, params={'iss.meta': 'off', 'limit': 1}, timeout=10)
        print(f"✓ API доступен. Статус: {response.status_code}")

        # Проверяем структуру ответа
        data = response.json()
        print(f"✓ Ответ получен. Ключи: {list(data.keys())}")

    except requests.exceptions.RequestException as e:
        print(f"✗ API недоступен: {e}")
        return False

    # Тест 2: Пробуем получить список бумаг
    print("\nПробуем получить список бумаг...")
    try:
        url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'securities',
            'securities.columns': 'SECID,SHORTNAME',
            'limit': 5
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'securities' in data:
            columns = data['securities']['columns']
            rows = data['securities']['data']
            print(f"✓ Данные получены. Бумаг: {len(rows)}")
            print(f"  Пример: {rows[0]}")
            return True
        else:
            print("✗ В ответе нет ключа 'securities'")
            print(f"  Ключи в ответе: {list(data.keys())}")
            return False

    except Exception as e:
        print(f"✗ Ошибка при получении бумаг: {e}")
        return False


if __name__ == "__main__":
    success = test_moex_api()
    if not success:
        print("\n❌ Проблема с доступом к MOEX API")