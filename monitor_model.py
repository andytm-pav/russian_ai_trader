#!/usr/bin/env python3
"""
Скрипт проверки реальной и ожидаемой размерности состояний в системе
Запуск: python check_state_dimensions.py
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')


def check_dimensions():
    print("=" * 80)
    print("ПРОВЕРКА РАЗМЕРНОСТЕЙ СОСТОЯНИЙ В СИСТЕМЕ")
    print("=" * 80)

    results = {
        'config_values': {},
        'model_values': {},
        'portfolio_values': {},
        'mismatches': []
    }

    # 1. Проверка конфигурационных файлов
    print("\n[1] КОНФИГУРАЦИОННЫЕ ФАЙЛЫ")
    print("-" * 40)

    try:
        with open("config/rl_config.json", "r", encoding="utf-8") as f:
            rl_config = json.load(f)

        state_params = rl_config.get('state_parameters', {})
        config_base_dim = state_params.get('state_vector_size', 'N/A')
        config_total_dim = state_params.get('total_state_size', 'N/A')
        config_news_dim = state_params.get('news_features_size', 'N/A')
        config_strategy_dim = state_params.get('strategy_params_size', 'N/A')

        print(f"rl_config.json:")
        print(f"  state_vector_size: {config_base_dim}")
        print(f"  total_state_size: {config_total_dim}")
        print(f"  news_features_size: {config_news_dim}")
        print(f"  strategy_params_size: {config_strategy_dim}")

        results['config_values'] = {
            'base_dim': config_base_dim,
            'total_dim': config_total_dim,
            'news_dim': config_news_dim,
            'strategy_dim': config_strategy_dim
        }

        # Проверка соответствия
        if config_base_dim != 'N/A' and config_strategy_dim != 'N/A':
            calculated_total = config_base_dim + config_strategy_dim
            if calculated_total != config_total_dim:
                mismatch = f"Конфиг: base({config_base_dim}) + strategy({config_strategy_dim}) = {calculated_total} ≠ total({config_total_dim})"
                results['mismatches'].append(mismatch)
                print(f"  ⚠️ НЕСООТВЕТСТВИЕ: {mismatch}")
            else:
                print(f"  ✅ Соответствует: {config_base_dim} + {config_strategy_dim} = {config_total_dim}")

    except Exception as e:
        print(f"  ❌ Ошибка загрузки rl_config.json: {e}")

    # 2. Проверка модели
    print("\n[2] ЗАГРУЖЕННАЯ МОДЕЛЬ")
    print("-" * 40)

    try:
        from models.trader_model import trader_model_instance, BASE_STATE_DIM, TOTAL_STATE_DIM, NEWS_ENCODED_DIM

        model = trader_model_instance

        model_base_dim = getattr(model, 'base_state_dim', 'N/A')
        model_total_dim = getattr(model, 'total_state_dim', 'N/A')
        model_news_dim = getattr(model, 'news_encoded_dim', 'N/A')
        model_strategy_dim = getattr(model, 'strategy_params_dim', 'N/A')

        print(f"Экземпляр модели:")
        print(f"  base_state_dim: {model_base_dim}")
        print(f"  total_state_dim: {model_total_dim}")
        print(f"  news_encoded_dim: {model_news_dim}")
        print(f"  strategy_params_dim: {model_strategy_dim}")
        print(f"\nГлобальные константы:")
        print(f"  BASE_STATE_DIM: {BASE_STATE_DIM}")
        print(f"  TOTAL_STATE_DIM: {TOTAL_STATE_DIM}")
        print(f"  NEWS_ENCODED_DIM: {NEWS_ENCODED_DIM}")

        results['model_values'] = {
            'base_dim': model_base_dim,
            'total_dim': model_total_dim,
            'news_dim': model_news_dim,
            'strategy_dim': model_strategy_dim,
            'global_base': BASE_STATE_DIM,
            'global_total': TOTAL_STATE_DIM
        }

        # Проверка размерности policy_net
        if hasattr(model, 'policy_net'):
            first_layer = None
            for name, module in model.policy_net.named_modules():
                if isinstance(module, torch.nn.Linear):
                    first_layer = module.in_features
                    break

            if first_layer:
                print(f"\n  PolicyNet входной слой: {first_layer}")
                results['model_values']['policy_net_input'] = first_layer

                if first_layer != model_total_dim:
                    mismatch = f"PolicyNet ожидает {first_layer}, модель говорит {model_total_dim}"
                    results['mismatches'].append(mismatch)
                    print(f"  ⚠️ НЕСООТВЕТСТВИЕ: {mismatch}")
                else:
                    print(f"  ✅ PolicyNet соответствует модели: {first_layer}")

        # Тест build_state_vector
        print(f"\n[3] ТЕСТ build_state_vector")
        print("-" * 40)

        test_ticker = "SBER"
        test_price = 300.0
        test_momentum = 0.02
        test_sentiment = 0.1
        test_news_features = torch.zeros(1, model_news_dim) if model_news_dim != 'N/A' else torch.zeros(1, 132)

        test_market_data = {
            'volume': 1e9, 'spread': 0.01, 'market_cap': 1e12,
            'rsi': 50, 'sma_10_ratio': 1.0, 'sma_20_ratio': 1.0,
            'bb_position': 0.5, 'atr': 3.0, 'volume_ratio': 1.0,
            'lot_size': 1, 'min_step': 0.01, 'sector': 'финансы',
            'momentum': test_momentum, 'imoex': 3000, 'imoex_change': 10,
            'rtsi': 1000, 'rtsi_change': 5, 'rvi': 20, 'rvi_change': 0,
            'moexog': 5000, 'moexfn': 8000, 'brent': 80, 'brent_change': 0.5,
            'market_liquidity_ratio': 0.5, 'market_activity_score': 0.7
        }

        try:
            state_vector = model.build_state_vector(
                ticker=test_ticker,
                price=test_price,
                momentum=test_momentum,
                sentiment=test_sentiment,
                news_features=test_news_features,
                market_data=test_market_data,
                market_sentiment=0.0,
                portfolio=None
            )

            actual_dim = state_vector.shape[0]
            print(f"  build_state_vector вернул: {actual_dim}")
            print(f"  Ожидалось (base_state_dim): {model_base_dim}")

            results['test_build'] = {
                'actual_dim': actual_dim,
                'expected_dim': model_base_dim
            }

            if actual_dim != model_base_dim:
                mismatch = f"build_state_vector возвращает {actual_dim}, ожидается {model_base_dim}"
                results['mismatches'].append(mismatch)
                print(f"  ⚠️ НЕСООТВЕТСТВИЕ: {mismatch}")
            else:
                print(f"  ✅ Размерность совпадает")

        except Exception as e:
            print(f"  ❌ Ошибка build_state_vector: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"  ❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()

    # 4. Проверка portfolio_state.json
    print("\n[4] СОХРАНЕННЫЕ СОСТОЯНИЯ В ПОРТФЕЛЕ")
    print("-" * 40)

    portfolio_path = Path("data/portfolio_state.json")
    if portfolio_path.exists():
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                portfolio = json.load(f)

            positions = portfolio.get('positions', {})
            print(f"  Позиций в портфеле: {len(positions)}")

            dimensions_found = {}
            for ticker, pos in positions.items():
                entry_state = pos.get('entry_state')
                if entry_state:
                    dim = len(entry_state)
                    dimensions_found[dim] = dimensions_found.get(dim, 0) + 1
                    strategy = pos.get('strategy', 'unknown')
                    print(f"    {ticker}: {dim} признаков (стратегия: {strategy})")

            if dimensions_found:
                print(f"\n  Распределение размерностей:")
                for dim, count in dimensions_found.items():
                    print(f"    {dim}: {count} позиций")

                results['portfolio_values'] = {
                    'dimensions': dimensions_found,
                    'total_positions': len(positions)
                }

                # Проверка на несоответствие
                expected = results['model_values'].get('base_dim', 210)
                for dim in dimensions_found.keys():
                    if dim != expected:
                        mismatch = f"В портфеле есть состояния размерности {dim}, ожидается {expected}"
                        results['mismatches'].append(mismatch)
                        print(f"\n  ⚠️ НЕСООТВЕТСТВИЕ: {mismatch}")
            else:
                print("  Нет сохраненных состояний")

        except Exception as e:
            print(f"  ❌ Ошибка чтения portfolio_state.json: {e}")
    else:
        print("  Файл portfolio_state.json не найден")

    # 5. Проверка ticker_states в памяти (если система запущена)
    print("\n[5] TICKER_STATES В ПАМЯТИ (если доступно)")
    print("-" * 40)

    try:
        from models.smart_broker import SmartPortfolioBroker
        import gc

        # Ищем экземпляр брокера в памяти
        broker_found = False
        for obj in gc.get_objects():
            if isinstance(obj, SmartPortfolioBroker):
                broker = obj
                broker_found = True

                if hasattr(broker, 'ticker_states'):
                    ticker_states = broker.ticker_states
                    print(f"  Найдено состояний в ticker_states: {len(ticker_states)}")

                    dims = {}
                    for ticker, state in ticker_states.items():
                        if isinstance(state, torch.Tensor):
                            dim = state.shape[0]
                            dims[dim] = dims.get(dim, 0) + 1
                            print(f"    {ticker}: {dim}")

                    if dims:
                        print(f"\n  Распределение в памяти:")
                        for dim, count in dims.items():
                            print(f"    {dim}: {count} тикеров")

                        results['ticker_states'] = {'dimensions': dims}

                        expected = results['model_values'].get('base_dim', 210)
                        for dim in dims.keys():
                            if dim != expected:
                                mismatch = f"В ticker_states есть состояния {dim}, ожидается {expected}"
                                results['mismatches'].append(mismatch)
                                print(f"\n  ⚠️ НЕСООТВЕТСТВИЕ: {mismatch}")
                    else:
                        print("  Нет состояний")
                else:
                    print("  ticker_states не найден в брокере")
                break

        if not broker_found:
            print("  Экземпляр SmartPortfolioBroker не найден в памяти (система не запущена)")

    except ImportError:
        print("  Не удалось импортировать SmartPortfolioBroker")
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")

    # ИТОГ
    print("\n" + "=" * 80)
    print("ИТОГ ПРОВЕРКИ")
    print("=" * 80)

    if results['mismatches']:
        print("\n❌ НАЙДЕНЫ НЕСООТВЕТСТВИЯ:")
        for i, mismatch in enumerate(results['mismatches'], 1):
            print(f"  {i}. {mismatch}")
        print("\n⚠️ Это является причиной зависаний!")
    else:
        print("\n✅ ВСЕ РАЗМЕРНОСТИ СООТВЕТСТВУЮТ")
        print("  Проблема не в размерностях состояний")

    # Сохраняем результаты
    output_path = Path("data/dimension_check.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nРезультаты сохранены в: {output_path}")

    return results


if __name__ == "__main__":
    check_dimensions()