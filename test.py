#!/usr/bin/env python3
"""
Проверка многопоточности и потенциальных deadlock
"""
import time
import threading
import queue
import sys
import tracemalloc

sys.path.insert(0, '.')


def test_thread_isolation():
    """Проверка изоляции потоков"""
    print("=" * 60)
    print("ТЕСТ МНОГОПОТОЧНОСТИ")
    print("=" * 60)

    results = {'main': None, 'thread1': None, 'thread2': None}
    errors = []

    def worker1():
        try:
            from models.trader_model import trader_model_instance
            import torch

            # Имитация выбора стратегии
            state = torch.randn(210)
            market_context = {
                'market_sentiment': 0.1,
                'volatility': 0.2,
                'confidence': 0.5,
                'time_of_day': 0.5,
                'ticker_sentiment': 0.0,
                'assigned_horizon': 'day'
            }

            start = time.time()
            action, strategy, conf = trader_model_instance.choose_action_with_strategy(
                state=state,
                ticker='TEST',
                price=100.0,
                market_context=market_context
            )
            elapsed = time.time() - start
            results['thread1'] = ('success', elapsed, strategy)
        except Exception as e:
            results['thread1'] = ('error', str(e))

    def worker2():
        try:
            from fetchers.moex_fetcher import MoexFetcher
            moex = MoexFetcher(use_cache=False)

            start = time.time()
            price = moex.get_price('SBER')
            elapsed = time.time() - start
            results['thread2'] = ('success', elapsed, price)
        except Exception as e:
            results['thread2'] = ('error', str(e))

    print("\nЗапуск параллельных потоков...")

    t1 = threading.Thread(target=worker1, name="ModelThread")
    t2 = threading.Thread(target=worker2, name="MOEXThread")

    start_all = time.time()
    t1.start()
    t2.start()

    t1.join(timeout=15)
    t2.join(timeout=15)
    total_time = time.time() - start_all

    print(f"\nОбщее время выполнения: {total_time:.3f}с")

    if t1.is_alive():
        print("❌ Поток 1 (модель) ЗАВИС!")
        errors.append("thread1_hang")
    else:
        status, elapsed, data = results['thread1']
        if status == 'success':
            print(f"✅ Поток 1: {elapsed:.3f}с, стратегия: {data}")
        else:
            print(f"❌ Поток 1 ошибка: {data}")

    if t2.is_alive():
        print("❌ Поток 2 (MOEX) ЗАВИС!")
        errors.append("thread2_hang")
    else:
        status, elapsed, data = results['thread2']
        if status == 'success':
            print(f"✅ Поток 2: {elapsed:.3f}с, цена SBER: {data}")
        else:
            print(f"❌ Поток 2 ошибка: {data}")

    return len(errors) == 0


def test_sequential_vs_parallel():
    """Сравнение последовательного и параллельного выполнения"""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПОСЛЕДОВАТЕЛЬНОГО И ПАРАЛЛЕЛЬНОГО ВЫПОЛНЕНИЯ")
    print("=" * 60)

    def model_task():
        from models.trader_model import trader_model_instance
        import torch
        state = torch.randn(210)
        market_context = {
            'market_sentiment': 0.1, 'volatility': 0.2, 'confidence': 0.5,
            'time_of_day': 0.5, 'ticker_sentiment': 0.0, 'assigned_horizon': 'day'
        }
        return trader_model_instance.choose_action_with_strategy(
            state=state, ticker='TEST', price=100.0, market_context=market_context
        )

    def moex_task():
        from fetchers.moex_fetcher import MoexFetcher
        moex = MoexFetcher(use_cache=False)
        return moex.get_price('SBER')

    # Последовательное выполнение
    print("\n1. ПОСЛЕДОВАТЕЛЬНО:")
    start = time.time()
    model_task()
    moex_task()
    sequential_time = time.time() - start
    print(f"   Время: {sequential_time:.3f}с")

    # Параллельное выполнение
    print("\n2. ПАРАЛЛЕЛЬНО:")

    result_queue = queue.Queue()

    def parallel_worker(task_func, name):
        try:
            start = time.time()
            task_func()
            elapsed = time.time() - start
            result_queue.put((name, 'success', elapsed))
        except Exception as e:
            result_queue.put((name, 'error', str(e)))

    start = time.time()
    t1 = threading.Thread(target=lambda: parallel_worker(model_task, "model"))
    t2 = threading.Thread(target=lambda: parallel_worker(moex_task, "moex"))
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)
    parallel_time = time.time() - start

    print(f"   Время: {parallel_time:.3f}с")

    if parallel_time > sequential_time * 3:
        print(f"\n⚠️ Параллельное выполнение ЗНАЧИТЕЛЬНО МЕДЛЕННЕЕ ({parallel_time:.3f}с vs {sequential_time:.3f}с)")
        print("   Это указывает на проблему с GIL или блокировками")
        return False
    else:
        print(f"\n✅ Параллельное выполнение в норме")
        return True


def test_news_fetcher_thread_safety():
    """Проверка потокобезопасности news_fetcher"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ПОТОКОБЕЗОПАСНОСТИ NEWS_FETCHER")
    print("=" * 60)

    from fetchers.news_fetcher import OptimizedNewsFetcher

    fetcher = OptimizedNewsFetcher("config/rss_sources.json")

    def news_worker(worker_id):
        try:
            start = time.time()
            news = fetcher.get_last_news(limit=10)
            elapsed = time.time() - start
            return ('success', worker_id, len(news), elapsed)
        except Exception as e:
            return ('error', worker_id, str(e))

    threads = []
    results_queue = queue.Queue()

    def worker_wrapper(wid):
        results_queue.put(news_worker(wid))

    print("\nЗапуск 5 параллельных запросов к news_fetcher...")

    start_all = time.time()
    for i in range(5):
        t = threading.Thread(target=lambda wid=i: worker_wrapper(wid))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=10)

    total_time = time.time() - start_all

    print(f"Общее время: {total_time:.3f}с\n")

    successes = 0
    while not results_queue.empty():
        status, wid, data, elapsed = results_queue.get()
        if status == 'success':
            successes += 1
            print(f"  Worker {wid}: {data} новостей за {elapsed:.3f}с")
        else:
            print(f"  Worker {wid}: ОШИБКА - {data}")

    if successes == 5:
        print(f"\n✅ Все 5 потоков отработали успешно")
        return True
    else:
        print(f"\n⚠️ Отработало только {successes}/5 потоков")
        return False


if __name__ == "__main__":
    print("ЗАПУСК ТЕСТОВ МНОГОПОТОЧНОСТИ\n")

    result1 = test_thread_isolation()
    result2 = test_sequential_vs_parallel()
    result3 = test_news_fetcher_thread_safety()

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)

    if result1 and result2 and result3:
        print("✅ Все тесты многопоточности пройдены")
        print("   Проблема НЕ в многопоточности")
        print("\n   Возможная причина: логика обработки сигналов в smart_broker.py")
        print("   Рекомендация: добавить подробное логирование в _execute_trading_decisions")
    else:
        print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ С МНОГОПОТОЧНОСТЬЮ")