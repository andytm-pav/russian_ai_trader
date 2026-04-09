#!/usr/bin/env python3
"""
Проверка MOEX API на таймауты и зависания
"""
import time
import threading
import queue
import sys

sys.path.insert(0, '.')


def test_moex_timeout():
    print("=" * 60)
    print("ТЕСТ MOEX API НА ТАЙМАУТЫ")
    print("=" * 60)

    from fetchers.moex_fetcher import MoexFetcher

    moex = MoexFetcher(use_cache=False)

    test_tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR', 'GMKN', 'NVTK', 'YNDX', 'TATN', 'PLZL']

    print(f"\nТестируем получение цен для {len(test_tickers)} тикеров...")
    print("(каждый запрос с таймаутом 5 секунд)\n")

    results = {}
    hangs = []

    for ticker in test_tickers:
        result_queue = queue.Queue()

        def get_price_worker():
            try:
                start = time.time()
                price = moex.get_price(ticker)
                elapsed = time.time() - start
                result_queue.put(('success', price, elapsed))
            except Exception as e:
                result_queue.put(('error', str(e)))

        thread = threading.Thread(target=get_price_worker, daemon=True)
        thread.start()
        thread.join(timeout=5)

        if thread.is_alive():
            print(f"  {ticker}: ⏰ ТАЙМАУТ (>5с) - ЗАВИСАНИЕ!")
            hangs.append(ticker)
            results[ticker] = {'status': 'HANG', 'time': 5.0}
        else:
            try:
                status, price, elapsed = result_queue.get_nowait()
                if status == 'success':
                    if price:
                        print(f"  {ticker}: {price:.2f}₽ ({elapsed:.3f}с)")
                        results[ticker] = {'status': 'OK', 'price': price, 'time': elapsed}
                    else:
                        print(f"  {ticker}: NULL ({elapsed:.3f}с)")
                        results[ticker] = {'status': 'NULL', 'time': elapsed}
                else:
                    print(f"  {ticker}: ОШИБКА - {price}")
                    results[ticker] = {'status': 'ERROR', 'error': price}
            except queue.Empty:
                print(f"  {ticker}: ⏰ ТАЙМАУТ (пустой результат)")
                hangs.append(ticker)
                results[ticker] = {'status': 'HANG', 'time': 5.0}

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    success_count = sum(1 for r in results.values() if r['status'] == 'OK' and r.get('price'))
    null_count = sum(1 for r in results.values() if r['status'] == 'NULL')
    error_count = sum(1 for r in results.values() if r['status'] == 'ERROR')
    hang_count = len(hangs)

    print(f"  Успешно: {success_count}")
    print(f"  NULL: {null_count}")
    print(f"  Ошибки: {error_count}")
    print(f"  Зависания: {hang_count}")

    if hang_count > 0:
        print(f"\n❌ ОБНАРУЖЕНЫ ЗАВИСАНИЯ MOEX API: {hangs}")
        return False
    else:
        print("\n✅ MOEX API работает без зависаний")
        return True


def test_moex_batch():
    """Тест батчевого запроса"""
    print("\n" + "=" * 60)
    print("ТЕСТ БАТЧЕВОГО ЗАПРОСА MOEX")
    print("=" * 60)

    from fetchers.moex_fetcher import MoexFetcher
    moex = MoexFetcher(use_cache=False)

    test_tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR']

    result_queue = queue.Queue()

    def batch_worker():
        try:
            start = time.time()
            prices = moex.get_prices_batch(test_tickers)
            elapsed = time.time() - start
            result_queue.put(('success', prices, elapsed))
        except Exception as e:
            result_queue.put(('error', str(e)))

    thread = threading.Thread(target=batch_worker, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        print("⏰ ТАЙМАУТ батчевого запроса (>10с) - ЗАВИСАНИЕ!")
        return False
    else:
        status, data, elapsed = result_queue.get_nowait()
        if status == 'success':
            print(f"✅ Батч выполнен за {elapsed:.3f}с")
            for ticker, price in data.items():
                print(f"  {ticker}: {price}")
            return True
        else:
            print(f"❌ Ошибка батча: {data}")
            return False


def test_moex_continuous():
    """Непрерывный тест MOEX (имитация работы системы)"""
    print("\n" + "=" * 60)
    print("НЕПРЕРЫВНЫЙ ТЕСТ MOEX (10 циклов)")
    print("=" * 60)

    from fetchers.moex_fetcher import MoexFetcher
    moex = MoexFetcher(use_cache=False)

    test_tickers = ['SBER', 'GAZP', 'LKOH']

    for cycle in range(10):
        print(f"\nЦикл {cycle + 1}:")

        result_queue = queue.Queue()

        def cycle_worker():
            try:
                prices = {}
                for ticker in test_tickers:
                    price = moex.get_price(ticker)
                    if price:
                        prices[ticker] = price
                result_queue.put(('success', prices))
            except Exception as e:
                result_queue.put(('error', str(e)))

        thread = threading.Thread(target=cycle_worker, daemon=True)
        thread.start()
        thread.join(timeout=5)

        if thread.is_alive():
            print(f"  ⏰ ТАЙМАУТ в цикле {cycle + 1}")
            return False
        else:
            status, data = result_queue.get_nowait()
            if status == 'success':
                print(f"  ✅ Получено {len(data)} цен")
            else:
                print(f"  ❌ Ошибка: {data}")

        time.sleep(1)

    return True


if __name__ == "__main__":
    print("ЗАПУСК ТЕСТОВ MOEX API\n")

    result1 = test_moex_timeout()
    result2 = test_moex_batch()
    result3 = test_moex_continuous()

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)

    if result1 and result2 and result3:
        print("✅ Все тесты MOEX пройдены")
    else:
        print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ С MOEX API")
        print("   Рекомендация: увеличить таймауты в moex_fetcher.py")