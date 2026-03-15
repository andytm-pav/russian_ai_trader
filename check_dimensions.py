"""
Скрипт для подсчета реальной размерности состояния из кода
Запускать после импорта trader_model
"""

import torch
from models.trader_model import trader_model_instance


def analyze_state_dimension():
    """Анализ размерности состояния"""

    print("=" * 60)
    print("АНАЛИЗ РАЗМЕРНОСТИ СОСТОЯНИЯ")
    print("=" * 60)

    # 1. Из конфига
    print("\n📁 ИЗ КОНФИГА:")
    print(f"  state_vector_size: {trader_model_instance.base_state_dim}")
    print(f"  total_state_size:  {trader_model_instance.total_state_dim}")

    # 2. Из архитектуры сети
    print("\n🧠 ИЗ АРХИТЕКТУРЫ СЕТИ:")
    print(f"  policy_net.state_dim: {trader_model_instance.policy_net.state_dim}")

    # 3. Создаем тестовое состояние и смотрим реальную размерность
    print("\n🔬 ТЕСТОВОЕ СОСТОЯНИЕ:")

    # Создаем заглушки для теста
    test_ticker = "SBER"
    test_price = 250.0
    test_momentum = 0.5
    test_sentiment = 0.1
    test_news = torch.zeros(1, trader_model_instance.news_encoded_dim).to(trader_model_instance.device)
    test_market_data = {
        'volume': 1000000,
        'spread': 0.01,
        'rsi': 55,
        'sma_10_ratio': 1.02,
        'sma_20_ratio': 1.01,
        'bb_position': 0.6,
        'atr': 2.5,
        'volume_ratio': 1.2,
        'market_cap': 1e12,
        'sector': 'финансы',
        'lot_size': 1,
        'min_step': 0.01,
    }

    # Создаем состояние
    test_state = trader_model_instance.build_state_vector(
        ticker=test_ticker,
        price=test_price,
        momentum=test_momentum,
        sentiment=test_sentiment,
        news_features=test_news,
        market_data=test_market_data,
        market_sentiment=0.0
    )

    print(f"  Реальная размерность build_state_vector: {test_state.shape[0]}")

    # 4. Проверка с добавлением стратегии
    print("\n🎯 С ДОБАВЛЕНИЕМ СТРАТЕГИИ:")

    strategy_params = trader_model_instance.strategies.get('balanced', {})
    full_state = trader_model_instance._create_strategy_state(test_state, strategy_params)

    print(f"  После добавления стратегии: {full_state.shape[0]}")
    print(f"  (добавлено {trader_model_instance.strategy_params_dim} параметров)")

    # 5. Вердикт
    print("\n✅ ВЕРДИКТ:")
    if test_state.shape[0] == trader_model_instance.base_state_dim:
        print(f"  ✓ base_state_dim совпадает: {test_state.shape[0]}")
    else:
        print(
            f"  ✗ НЕСОВПАДЕНИЕ! base_state_dim={trader_model_instance.base_state_dim}, реальная={test_state.shape[0]}")

    if full_state.shape[0] == trader_model_instance.total_state_dim:
        print(f"  ✓ total_state_dim совпадает: {full_state.shape[0]}")
    else:
        print(
            f"  ✗ НЕСОВПАДЕНИЕ! total_state_dim={trader_model_instance.total_state_dim}, реальная={full_state.shape[0]}")

    print("=" * 60)

    return {
        'config_base': trader_model_instance.base_state_dim,
        'config_total': trader_model_instance.total_state_dim,
        'real_base': test_state.shape[0],
        'real_total': full_state.shape[0],
        'strategy_params': trader_model_instance.strategy_params_dim
    }


def analyze_build_state_vector_code():
    """Анализ кода build_state_vector для подсчета размерности"""

    print("\n📝 АНАЛИЗ КОДА build_state_vector:")
    print("  Группы признаков:")

    # Эти числа нужно сверить с реальным кодом
    dimensions = {
        'price_volume': 4,
        'technical': 7,
        'news_base': 128,
        'news_reserve': 2,
        'fundamental': 7,
        'position': 5,
        'portfolio': 5,
        'risk': 3,
        'time': 4,
        'strategy': 11,
        'macro_reserve': 10
    }

    total = sum(dimensions.values())

    for name, size in dimensions.items():
        print(f"    {name:20} : {size:3}")

    print(f"\n    {'ИТОГО':20} : {total:3}")

    return dimensions


if __name__ == "__main__":
    print("Запуск анализа размерности...")
    result = analyze_state_dimension()
    dimensions = analyze_build_state_vector_code()