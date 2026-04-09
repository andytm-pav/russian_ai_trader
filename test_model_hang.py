#!/usr/bin/env python3
"""
Скрипт проверки зависания нейросети
Запускать во время нормальной работы или когда система перестала торговать
"""

import json
import time
import sys
import threading
import queue
import torch
import traceback
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, '.')


def test_model_responsiveness(timeout_seconds=5):
    """Проверка отзывчивости модели"""
    print(f"\n{'=' * 60}")
    print(f"ДИАГНОСТИКА МОДЕЛИ - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 60}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'overall_status': 'UNKNOWN'
    }

    # Тест 1: Импорт модели
    print("\n[1/6] Импорт модели...")
    try:
        start = time.time()
        from models.trader_model import trader_model_instance
        import_time = time.time() - start
        print(f"      ✅ Успешно за {import_time:.3f}с")
        results['tests']['import'] = {'status': 'OK', 'time': import_time}
        model = trader_model_instance
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
        results['tests']['import'] = {'status': 'FAIL', 'error': str(e)}
        results['overall_status'] = 'FAIL'
        return results

    # Тест 2: Состояние модели
    print("\n[2/6] Проверка состояния модели...")
    try:
        model_info = {
            'device': str(model.device),
            'model_loaded': hasattr(model, 'policy_net'),
            'state_dim': getattr(model, 'state_dim', 'N/A'),
            'action_dim': getattr(model, 'action_dim', 'N/A'),
            'memory_size': len(model.memory) if hasattr(model, 'memory') else 0,
            'prioritized_buffer_size': model.prioritized_buffer.size if hasattr(model, 'prioritized_buffer') else 0
        }
        print(f"      Device: {model_info['device']}")
        print(f"      Policy net: {'✅' if model_info['model_loaded'] else '❌'}")
        print(f"      State dim: {model_info['state_dim']}")
        print(f"      Memory size: {model_info['memory_size']}")
        results['tests']['state'] = {'status': 'OK', 'info': model_info}
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
        results['tests']['state'] = {'status': 'FAIL', 'error': str(e)}

    # Тест 3: Простой forward pass
    print("\n[3/6] Тест forward pass (с таймаутом)...")

    result_queue = queue.Queue()

    def forward_test():
        try:
            start = time.time()
            # Создаем тестовый тензор
            state_dim = getattr(model, 'state_dim', 150)
            test_state = torch.randn(1, state_dim, device=model.device)

            # Переводим в eval mode
            model.policy_net.eval()

            with torch.no_grad():
                probs, value = model.policy_net(test_state)

            elapsed = time.time() - start
            result_queue.put({
                'status': 'OK',
                'time': elapsed,
                'probs_shape': list(probs.shape),
                'value_shape': list(value.shape)
            })
        except Exception as e:
            result_queue.put({'status': 'ERROR', 'error': str(e), 'traceback': traceback.format_exc()})

    thread = threading.Thread(target=forward_test, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"      ⏰ ТАЙМАУТ! Модель зависла (> {timeout_seconds}с)")
        results['tests']['forward_pass'] = {'status': 'HANG', 'timeout': timeout_seconds}
        results['overall_status'] = 'HANG'
    else:
        try:
            result = result_queue.get_nowait()
            if result['status'] == 'OK':
                print(f"      ✅ Успешно за {result['time']:.3f}с")
                print(f"         Probs: {result['probs_shape']}, Value: {result['value_shape']}")
                results['tests']['forward_pass'] = result
            else:
                print(f"      ❌ Ошибка: {result['error']}")
                results['tests']['forward_pass'] = result
        except queue.Empty:
            print(f"      ❌ Пустой результат")
            results['tests']['forward_pass'] = {'status': 'EMPTY'}

    # Тест 4: choose_action_with_strategy
    print("\n[4/6] Тест choose_action_with_strategy (с таймаутом)...")

    result_queue = queue.Queue()

    def strategy_test():
        try:
            start = time.time()

            state_dim = getattr(model, 'state_dim', 150)
            test_state = torch.randn(state_dim, device=model.device)

            market_context = {
                'market_sentiment': 0.1,
                'volatility': 0.2,
                'confidence': 0.5,
                'time_of_day': 0.5,
                'ticker_sentiment': 0.0,
                'assigned_horizon': 'day'
            }

            action, strategy, confidence = model.choose_action_with_strategy(
                state=test_state,
                ticker='TEST',
                price=100.0,
                market_context=market_context
            )

            elapsed = time.time() - start
            result_queue.put({
                'status': 'OK',
                'time': elapsed,
                'action': action,
                'strategy': strategy,
                'confidence': confidence
            })
        except Exception as e:
            result_queue.put({'status': 'ERROR', 'error': str(e), 'traceback': traceback.format_exc()})

    thread = threading.Thread(target=strategy_test, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        print(f"      ⏰ ТАЙМАУТ! choose_action_with_strategy завис (> {timeout_seconds}с)")
        results['tests']['choose_action'] = {'status': 'HANG', 'timeout': timeout_seconds}
        results['overall_status'] = 'HANG'
    else:
        try:
            result = result_queue.get_nowait()
            if result['status'] == 'OK':
                print(f"      ✅ Успешно за {result['time']:.3f}с")
                print(
                    f"         Action: {result['action']}, Strategy: {result['strategy']}, Conf: {result['confidence']:.3f}")
                results['tests']['choose_action'] = result
            else:
                print(f"      ❌ Ошибка: {result['error']}")
                results['tests']['choose_action'] = result
        except queue.Empty:
            print(f"      ❌ Пустой результат")
            results['tests']['choose_action'] = {'status': 'EMPTY'}

    # Тест 5: Многократные вызовы
    print("\n[5/6] Тест стабильности (10 вызовов)...")
    try:
        times = []
        state_dim = getattr(model, 'state_dim', 150)

        for i in range(10):
            start = time.time()
            test_state = torch.randn(1, state_dim, device=model.device)
            with torch.no_grad():
                model.policy_net(test_state)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        print(f"      ✅ Среднее: {avg_time * 1000:.2f}ms, Макс: {max_time * 1000:.2f}ms")
        results['tests']['stability'] = {
            'status': 'OK',
            'avg_ms': avg_time * 1000,
            'max_ms': max_time * 1000,
            'times_ms': [t * 1000 for t in times]
        }
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
        results['tests']['stability'] = {'status': 'FAIL', 'error': str(e)}

    # Тест 6: Проверка памяти GPU
    print("\n[6/6] Проверка памяти...")
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"      GPU Allocated: {allocated:.2f} GB")
            print(f"      GPU Reserved: {reserved:.2f} GB")
            results['tests']['memory'] = {
                'status': 'OK',
                'gpu_allocated_gb': allocated,
                'gpu_reserved_gb': reserved
            }
        else:
            print(f"      CPU mode")
            results['tests']['memory'] = {'status': 'OK', 'mode': 'CPU'}
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
        results['tests']['memory'] = {'status': 'FAIL', 'error': str(e)}

    # Итог
    if results['overall_status'] == 'UNKNOWN':
        results['overall_status'] = 'OK'

    print(f"\n{'=' * 60}")
    print(f"ИТОГ: {results['overall_status']}")
    print(f"{'=' * 60}")

    # Сохраняем результаты
    with open('data/model_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def monitor_continuously(interval_seconds=60):
    """Непрерывный мониторинг"""
    print("Запуск непрерывного мониторинга модели...")
    print(f"Интервал проверки: {interval_seconds}с")
    print("Нажмите Ctrl+C для остановки\n")

    hang_count = 0

    try:
        while True:
            results = test_model_responsiveness(timeout_seconds=5)

            if results['overall_status'] == 'HANG':
                hang_count += 1
                print(f"\n⚠️ МОДЕЛЬ ЗАВИСЛА! (срабатывание #{hang_count})")

                if hang_count >= 3:
                    print("\n❌ КРИТИЧЕСКОЕ ЗАВИСАНИЕ! ТРЕБУЕТСЯ ПЕРЕЗАПУСК!")
                    break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\nМониторинг остановлен")


def check_current_state():
    """Проверка текущего состояния (для вызова из работающей системы)"""
    try:
        from models.trader_model import trader_model_instance

        # Проверяем, не заблокирована ли модель
        start = time.time()
        test_tensor = torch.randn(1, 150, device=trader_model_instance.device)

        with torch.no_grad():
            _ = trader_model_instance.policy_net(test_tensor)

        elapsed = time.time() - start

        if elapsed > 2.0:
            return {'status': 'SLOW', 'time': elapsed}
        else:
            return {'status': 'OK', 'time': elapsed}

    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Диагностика зависания модели')
    parser.add_argument('--monitor', action='store_true', help='Непрерывный мониторинг')
    parser.add_argument('--interval', type=int, default=60, help='Интервал мониторинга в секундах')
    parser.add_argument('--once', action='store_true', help='Однократная проверка')
    parser.add_argument('--timeout', type=int, default=5, help='Таймаут в секундах')

    args = parser.parse_args()

    if args.monitor:
        monitor_continuously(interval_seconds=args.interval)
    elif args.once:
        test_model_responsiveness(timeout_seconds=args.timeout)
    else:
        # По умолчанию - однократная проверка
        test_model_responsiveness(timeout_seconds=args.timeout)