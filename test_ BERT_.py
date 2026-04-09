#!/usr/bin/env python3
"""
Проверка BERT на зависание
"""
import time
import threading
import queue
import sys

sys.path.insert(0, '.')


def test_bert_hang():
    print("=" * 60)
    print("ТЕСТ BERT НА ЗАВИСАНИЕ")
    print("=" * 60)

    from models.trader_model import trader_model_instance

    model = trader_model_instance

    if model.bert_model is None:
        print("❌ BERT не загружен")
        return

    # Тестовые тексты
    test_texts = [
        "Сбербанк объявил о выплате дивидендов",
        "Нефть Brent подорожала до 85 долларов за баррель",
        "Рынок акций РФ закрылся ростом",
        "ЦБ РФ сохранил ключевую ставку",
        "Газпром увеличил добычу газа"
    ]

    print(f"\nТестируем кодирование {len(test_texts)} текстов...")

    result_queue = queue.Queue()

    def encode_worker():
        try:
            start = time.time()
            features = model.encode_news(test_texts)
            elapsed = time.time() - start
            result_queue.put(('success', elapsed, features.shape))
        except Exception as e:
            result_queue.put(('error', str(e)))

    thread = threading.Thread(target=encode_worker, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if thread.is_alive():
        print("\n⏰ ТАЙМАУТ! BERT завис при кодировании (>10 секунд)")
        print("❌ ПРИЧИНА ЗАВИСАНИЯ НАЙДЕНА: BERT блокируется")
        return False
    else:
        status, *data = result_queue.get_nowait()
        if status == 'success':
            elapsed, shape = data
            print(f"✅ BERT отработал за {elapsed:.3f}с, выходная форма: {shape}")
            return True
        else:
            print(f"❌ Ошибка BERT: {data[0]}")
            return False


def test_multiple_bert_calls():
    """Множественные вызовы BERT"""
    print("\n" + "=" * 60)
    print("ТЕСТ МНОЖЕСТВЕННЫХ ВЫЗОВОВ BERT (10 раз)")
    print("=" * 60)

    from models.trader_model import trader_model_instance
    model = trader_model_instance

    test_texts = ["Тестовая новость номер {}".format(i) for i in range(5)]

    times = []
    for i in range(10):
        start = time.time()
        try:
            features = model.encode_news(test_texts)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Вызов {i + 1}: {elapsed:.3f}с, форма: {features.shape}")
        except Exception as e:
            print(f"  Вызов {i + 1}: ОШИБКА - {e}")
            return False

    avg = sum(times) / len(times)
    max_time = max(times)
    print(f"\nСреднее: {avg:.3f}с, Максимум: {max_time:.3f}с")

    if max_time > 5.0:
        print("⚠️ BERT периодически замедляется (>5с)")
        return False
    return True


if __name__ == "__main__":
    result1 = test_bert_hang()
    print()
    result2 = test_multiple_bert_calls()

    print("\n" + "=" * 60)
    if result1 and result2:
        print("✅ BERT работает стабильно, проблема не в нем")
    else:
        print("❌ ОБНАРУЖЕНА ПРОБЛЕМА С BERT")
    print("=" * 60)