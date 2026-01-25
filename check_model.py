import json
import torch
from models.trader_model import trader_model_instance
from utils.logger import setup_logger

logger = setup_logger("MODEL_CHECK")


def check_model_status():
    """Визуальная проверка статуса модели"""

    model = trader_model_instance
    print("=" * 60)
    print("СТАТУС НЕЙРОСЕТЕВОЙ МОДЕЛИ ТРЕЙДЕРА")
    print("=" * 60)

    # 1. Базовая информация
    print(f"\n📊 ОСНОВНАЯ ИНФОРМАЦИЯ:")
    print(f"   Устройство: {model.device}")
    print(f"   Путь модели: {model.model_dir}")

    # 2. Память и опыт
    print(f"\n🧠 ПАМЯТЬ И ОПЫТ:")
    print(f"   Размер памяти: {len(model.memory)} / {model.memory.maxlen}")
    print(f"   Уникальных тикеров: {len(model.ticker_stats)}")
    print(f"   Ошибок в памяти: {len(model.error_memory)}")

    # 3. Статистика обучения
    print(f"\n📈 СТАТИСТИКА ОБУЧЕНИЯ:")

    # Проверяем файл состояния
    state_path = f"{model.model_dir}/model_state.json"
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        print(f"   Файл состояния: {state_path}")
        print(f"   Размер файла: {os.path.getsize(state_path) / 1024:.1f} KB")
        print(f"   Время сохранения: {state.get('save_time', 'неизвестно')}")

        # Анализ памяти
        memory_size = state.get('memory_size', 0)
        total_experiences = state.get('total_experiences', 0)

        print(f"   Всего опытов: {total_experiences}")
        print(f"   Ошибок тикеров: {len(state.get('error_memory', {}))}")

    except FileNotFoundError:
        print(f"   ❌ Файл состояния не найден: {state_path}")
    except Exception as e:
        print(f"   ⚠ Ошибка загрузки состояния: {e}")

    # 4. Рыночные метрики
    print(f"\n📊 РЫНОЧНЫЕ МЕТРИКИ:")
    print(f"   Рыночный сентимент: {model.market_sentiment:.3f}")
    print(f"   Индекс волатильности: {model.volatility_index:.2f}")

    # 5. Стратегии
    print(f"\n🎯 СТРАТЕГИИ:")
    strategies = getattr(model, 'strategies', {})
    print(f"   Доступно стратегий: {len(strategies)}")

    # Показываем статистику по стратегиям
    for strategy_name, params in list(strategies.items())[:3]:
        perf = model.strategy_performance.get(strategy_name, {})
        total_trades = perf.get('total_trades', 0)
        win_rate = perf.get('win_rate', 0)

        if total_trades > 0:
            print(f"   {strategy_name}: {total_trades} сделок, Win Rate: {win_rate:.1%}")

    # 6. Проверка нейросети
    print(f"\n🔧 ПРОВЕРКА НЕЙРОСЕТИ:")

    # Проверяем загружены ли веса
    weights_path = f"{model.model_dir}/model_weights.pth"
    if os.path.exists(weights_path):
        print(f"   ✅ Веса нейросети загружены ({os.path.getsize(weights_path) / 1024:.1f} KB)")

        # Проверяем параметры
        total_params = sum(p.numel() for p in model.policy_net.parameters())
        trainable_params = sum(p.numel() for p in model.policy_net.parameters() if p.requires_grad)

        print(f"   Параметры сети: {total_params:,}")
        print(f"   Обучаемых параметров: {trainable_params:,}")
    else:
        print(f"   ⚠ Веса нейросети не найдены")

    # 7. Тестовый прогон
    print(f"\n🧪 ТЕСТОВЫЙ ПРОГОН:")

    if len(model.memory) > 0:
        # Берем последнее состояние из памяти
        last_exp = list(model.memory)[-1]

        # Проверяем архитектуру
        state_shape = last_exp['state'].shape if hasattr(last_exp['state'], 'shape') else 'неизвестно'
        print(f"   Размер состояния: {state_shape}")
        print(f"   Последнее действие: {last_exp.get('action', 'неизвестно')}")
        print(f"   Награда: {last_exp.get('reward', 0):.3f}")
    else:
        print(f"   ℹ Память пуста")

    # 8. Оценка обученности
    print(f"\n📊 ОЦЕНКА ОБУЧЕННОСТИ:")

    score = 0
    max_score = 8

    # Критерии
    if len(model.memory) > 100:
        print("   ✅ Память достаточно наполнена")
        score += 2
    else:
        print("   ⚠ Мало данных в памяти")

    if len(model.ticker_stats) > 5:
        print("   ✅ Есть статистика по тикерам")
        score += 2
    else:
        print("   ⚠ Мало тикеров в статистике")

    if os.path.exists(weights_path):
        print("   ✅ Веса модели сохранены")
        score += 2
    else:
        print("   ⚠ Веса не сохранены")

    if hasattr(model, 'market_sentiment'):
        print("   ✅ Рыночный сентимент активен")
        score += 2
    else:
        print("   ⚠ Рыночный сентимент не настроен")

    # Итоговая оценка
    percent = (score / max_score) * 100
    print(f"\n📈 ИТОГОВАЯ ОЦЕНКА: {score}/{max_score} ({percent:.0f}%)")

    if percent > 75:
        print("   ✅ Модель хорошо обучена")
    elif percent > 50:
        print("   ⚠ Модель частично обучена")
    else:
        print("   ❌ Модель требует обучения")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import os

    check_model_status()