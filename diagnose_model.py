"""
Диагностика состояния модели TraderModel
Оценка готовности к продакшену
"""

import json
import numpy as np
import torch
from collections import deque
from datetime import datetime
import pickle
import gzip


def diagnose_trader_model(model_instance=None, memory_file="models/saved_trader/memory_buffer.pkl"):
    """
    Комплексная диагностика модели трейдера
    Оценка готовности к продакшену в процентах
    """

    print("=" * 80)
    print("ДИАГНОСТИКА TRADER MODEL")
    print("=" * 80)

    # Подсчет баллов для оценки готовности
    total_score = 0
    max_score = 100
    issues = []

    # 1. ПРОВЕРКА КОНФИГУРАЦИИ
    print("\n1. КОНФИГУРАЦИЯ:")
    print("-" * 40)

    try:
        with open("config/rl_config.json", "r") as f:
            rl_config = json.load(f)
            print(f"   reward_scaling = {rl_config.get('reward_scaling', 'N/A')}")
            print(f"   reward_clip_min = {rl_config.get('reward_clip_min', 'N/A')}")
            print(f"   reward_clip_max = {rl_config.get('reward_clip_max', 'N/A')}")

            # Проверка наличия всех параметров
            required_params = ['reward_scaling', 'price_change_threshold', 'learning_rate']
            missing = [p for p in required_params if p not in rl_config]
            if missing:
                issues.append(f"Отсутствуют параметры в конфиге: {missing}")
                total_score -= 10
            else:
                total_score += 5
    except Exception as e:
        issues.append(f"Ошибка загрузки конфига: {e}")
        total_score -= 15

    # 2. ПАМЯТЬ МОДЕЛИ
    print("\n2. ПАМЯТЬ МОДЕЛИ:")
    print("-" * 40)

    try:
        if model_instance and hasattr(model_instance, 'memory'):
            memory = model_instance.memory
        else:
            with gzip.open(memory_file, 'rb') as f:
                memory = pickle.load(f)

        memory_size = len(memory)
        print(f"   Размер памяти: {memory_size} опытов")

        # Анализ качества памяти
        if memory_size > 1000:
            print(f"   ✅ Отлично: >1000 опытов")
            total_score += 20
        elif memory_size > 500:
            print(f"   👍 Хорошо: 500-1000 опытов")
            total_score += 15
        elif memory_size > 100:
            print(f"   ⚠️ Приемлемо: 100-500 опытов")
            total_score += 10
        else:
            print(f"   ❌ Критично: <100 опытов")
            issues.append("Недостаточно опытов для обучения")
            total_score -= 20

        # Анализ reward
        if memory_size > 0:
            rewards = [exp['reward'] for exp in memory if 'reward' in exp]
            if rewards:
                avg_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                print(f"   Средний reward: {avg_reward:.4f} (σ={std_reward:.4f})")

                # Проверка масштаба reward
                if abs(avg_reward) < 1.0:
                    print(f"   ✅ Reward в правильном масштабе")
                    total_score += 15
                else:
                    print(f"   ❌ Reward слишком большие: {avg_reward:.2f}")
                    issues.append("Неправильный масштаб reward")
                    total_score -= 15
    except Exception as e:
        print(f"   Ошибка загрузки памяти: {e}")
        issues.append("Нет данных для обучения")
        total_score -= 30

    # 3. СТАТИСТИКА ТИКЕРОВ
    print("\n3. СТАТИСТИКА ТИКЕРОВ:")
    print("-" * 40)

    if model_instance and hasattr(model_instance, 'ticker_stats'):
        ticker_stats = model_instance.ticker_stats
        active_tickers = len(ticker_stats)
        print(f"   Тикеров в статистике: {active_tickers}")

        # Считаем тикеры с достаточной статистикой
        trained_tickers = sum(1 for stats in ticker_stats.values()
                              if stats.get('total_trades', 0) >= 10)
        print(f"   Тикеров с >10 сделками: {trained_tickers}")

        if trained_tickers > 20:
            total_score += 15
        elif trained_tickers > 10:
            total_score += 10
        elif trained_tickers > 5:
            total_score += 5
    else:
        print("   Нет данных по тикерам")
        issues.append("Нет статистики по тикерам")
        total_score -= 10

    # 4. ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ
    print("\n4. ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ:")
    print("-" * 40)

    if model_instance and hasattr(model_instance, 'strategy_performance'):
        strategy_stats = model_instance.strategy_performance

        for strategy, perf in strategy_stats.items():
            trades = perf.get('total_trades', 0)
            win_rate = perf.get('win_rate', 0)
            if trades > 0:
                print(f"   {strategy}: trades={trades}, win_rate={win_rate:.1%}")

                if win_rate > 0.55:
                    total_score += 5
                elif win_rate < 0.45 and trades > 20:
                    issues.append(f"Стратегия {strategy} убыточна")
                    total_score -= 5

    # 5. ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ
    print("\n5. ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ:")
    print("-" * 40)

    if model_instance:
        # Размерность состояния
        if hasattr(model_instance.policy_net, 'state_dim'):
            state_dim = model_instance.policy_net.state_dim
            print(f"   Размерность состояния: {state_dim}")
            if state_dim == 156:
                total_score += 5

        # Device
        device = model_instance.device
        print(f"   Устройство: {device}")
        if device.type == 'cuda':
            total_score += 5

        # BERT модель
        if model_instance.bert_model is not None:
            print(f"   ✅ BERT модель загружена")
            total_score += 10
        else:
            print(f"   ⚠️ BERT не загружен (fallback)")
            issues.append("Используется упрощенный анализ новостей")

    # 6. РЫНОЧНЫЕ ПОКАЗАТЕЛИ
    print("\n6. РЫНОЧНЫЕ ПОКАЗАТЕЛИ:")
    print("-" * 40)

    if model_instance:
        market_sentiment = model_instance.market_sentiment
        volatility = model_instance.volatility_index

        print(f"   Рыночный сентимент: {market_sentiment:.3f}")
        print(f"   Индекс волатильности: {volatility:.3f}")

        # Нормальные диапазоны
        if -0.5 < market_sentiment < 0.5:
            total_score += 5

        if 0.5 < volatility < 2.0:
            total_score += 5

    # 7. ОЦЕНКА ГОТОВНОСТИ К ПРОДАКШЕНУ
    print("\n" + "=" * 80)
    print("ОЦЕНКА ГОТОВНОСТИ К ПРОДАКШЕНУ")
    print("=" * 80)

    # Корректировка баллов (не ниже 0, не выше 100)
    final_score = max(0, min(100, total_score))

    print(f"\nИТОГОВАЯ ОЦЕНКА: {final_score}%")

    # Расшифровка оценки
    if final_score >= 90:
        print("\n🏆 СТАТУС: PRODUCTION READY")
        print("   Модель полностью готова к реальной торговле")
        print("   Рекомендации: регулярный мониторинг, A/B тестирование")
    elif final_score >= 75:
        print("\n✅ СТАТУС: READY WITH CAUTION")
        print("   Можно использовать, но с ограничениями")
        print("   Рекомендации: бумажная торговля, малый капитал")
    elif final_score >= 50:
        print("\n⚠️ СТАТУС: DEVELOPMENT")
        print("   Модель требует доработки")
        print("   Рекомендации: сбор данных, оптимизация параметров")
    else:
        print("\n❌ СТАТУС: EXPERIMENTAL")
        print("   Модель не готова к продакшену")
        print("   Рекомендации: исправление критических проблем")

    # Список проблем
    if issues:
        print("\nПРОБЛЕМЫ ДЛЯ РЕШЕНИЯ:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

    # Итоговые рекомендации
    print("\nРЕКОМЕНДАЦИИ:")
    if final_score < 50:
        print("   1. Соберите больше данных (минимум 500 опытов)")
        print("   2. Проверьте конфигурацию reward_scaling")
        print("   3. Добавьте BERT модель для новостей")
        print("   4. Оптимизируйте гиперпараметры")
    elif final_score < 75:
        print("   1. Увеличьте размер памяти до 1000+ опытов")
        print("   2. Протестируйте на бумажном счете 1 месяц")
        print("   3. Добавьте мониторинг в реальном времени")
    else:
        print("   1. Подготовьте систему мониторинга")
        print("   2. Настройте алерты на аномалии")
        print("   3. Запланируйте регулярное переобучение")

    return final_score


if __name__ == "__main__":
    # Попытка загрузить модель
    try:
        from models.trader_model import trader_model_instance

        score = diagnose_trader_model(model_instance=trader_model_instance)
    except:
        # Если модель не загружается, диагностируем по файлам
        score = diagnose_trader_model()