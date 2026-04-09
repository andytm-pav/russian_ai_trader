#!/usr/bin/env python3
"""
Комплексная диагностика модели TraderModel
Проверяет:
- Загрузку модели и конфигов
- Размерности и целостность весов
- Память и приоритетный буфер
- Forward pass и выбор стратегии
- Сохраненные состояния
"""

import sys
import json
import time
import traceback
from pathlib import Path

import torch
import numpy as np

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent))


# ======================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ========================

def check_tensor(tensor, name="tensor"):
    """Проверка тензора на NaN/Inf и корректность"""
    if tensor is None:
        return False, f"{name} is None"

    if not isinstance(tensor, torch.Tensor):
        return False, f"{name} is not a Tensor"

    if torch.isnan(tensor).any():
        return False, f"{name} contains NaN"

    if torch.isinf(tensor).any():
        return False, f"{name} contains Inf"

    return True, f"{name} OK, shape={tensor.shape}, dtype={tensor.dtype}"


def check_module_weights(module, module_name="Module"):
    """Проверка весов модуля на NaN/Inf и градиенты"""
    issues = []
    for name, param in module.named_parameters():
        if param is None:
            issues.append(f"{module_name}.{name} is None")
            continue
        if torch.isnan(param).any():
            issues.append(f"{module_name}.{name} contains NaN")
        if torch.isinf(param).any():
            issues.append(f"{module_name}.{name} contains Inf")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                issues.append(f"{module_name}.{name}.grad contains NaN")
            if torch.isinf(param.grad).any():
                issues.append(f"{module_name}.{name}.grad contains Inf")
    return issues


def format_size(size_bytes):
    """Форматирование размера в читаемый вид"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


# ======================== ОСНОВНАЯ ДИАГНОСТИКА ========================

def run_diagnostics():
    print("=" * 80)
    print("КОМПЛЕКСНАЯ ДИАГНОСТИКА МОДЕЛИ TRADERMODEL")
    print("=" * 80)

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checks': {},
        'warnings': [],
        'errors': []
    }

    # ---------- 1. Импорт модели ----------
    print("\n[1/8] Импорт модели и конфигов...")
    try:
        start = time.time()
        from models.trader_model import (
            trader_model_instance,
            NEWS_ENCODED_DIM,
            BASE_STATE_DIM,
            TOTAL_STATE_DIM
        )
        import_time = time.time() - start
        print(f"  ✅ Модель импортирована за {import_time:.3f}с")
        results['checks']['import'] = {'status': 'OK', 'time': import_time}

        model = trader_model_instance

    except Exception as e:
        print(f"  ❌ Ошибка импорта: {e}")
        traceback.print_exc()
        results['checks']['import'] = {'status': 'FAIL', 'error': str(e)}
        return results

    # ---------- 2. Проверка конфигурации ----------
    print("\n[2/8] Проверка конфигурации...")
    try:
        config_checks = {}

        # Проверка наличия конфигов
        config_files = [
            'config/rl_config.json',
            'config/strategies.json',
            'config/market_schedule.json'
        ]
        for cf in config_files:
            exists = Path(cf).exists()
            config_checks[cf] = 'exists' if exists else 'missing'
            if not exists:
                results['warnings'].append(f"Отсутствует {cf}")

        # Проверка загруженных параметров
        expected_attrs = [
            'base_state_dim', 'total_state_dim', 'news_encoded_dim',
            'strategy_params_dim', 'device', 'gamma', 'exploration_rate'
        ]
        for attr in expected_attrs:
            val = getattr(model, attr, None)
            config_checks[attr] = val

        print(f"  base_state_dim: {model.base_state_dim}")
        print(f"  total_state_dim: {model.total_state_dim}")
        print(f"  news_encoded_dim: {model.news_encoded_dim}")
        print(f"  device: {model.device}")
        print(f"  strategies: {len(model.strategies)} шт.")

        # Проверка соответствия размерностей
        if model.total_state_dim != model.base_state_dim + model.strategy_params_dim:
            msg = f"Несоответствие: total({model.total_state_dim}) != base({model.base_state_dim}) + strategy({model.strategy_params_dim})"
            results['errors'].append(msg)
            print(f"  ⚠️ {msg}")
        else:
            print(f"  ✅ Размерности согласованы")

        results['checks']['config'] = {'status': 'OK', 'details': config_checks}

    except Exception as e:
        print(f"  ❌ Ошибка проверки конфигурации: {e}")
        results['checks']['config'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- 3. Проверка нейронных сетей ----------
    print("\n[3/8] Проверка нейронных сетей...")
    try:
        network_checks = {}

        # NewsEncoder
        print("  NewsEncoder:")
        issues = check_module_weights(model.news_encoder, "NewsEncoder")
        if issues:
            for iss in issues:
                print(f"    ⚠️ {iss}")
                results['warnings'].append(iss)
        else:
            print("    ✅ Веса в порядке")

        # PolicyNet
        print("  PolicyNet:")
        issues = check_module_weights(model.policy_net, "PolicyNet")
        if issues:
            for iss in issues:
                print(f"    ⚠️ {iss}")
                results['warnings'].append(iss)
        else:
            print("    ✅ Веса в порядке")

        # Проверка входного слоя PolicyNet
        first_linear = None
        for module in model.policy_net.modules():
            if isinstance(module, torch.nn.Linear):
                first_linear = module
                break
        if first_linear:
            input_dim = first_linear.in_features
            print(f"    Входной слой ожидает: {input_dim}")
            if input_dim != model.total_state_dim:
                msg = f"PolicyNet ожидает {input_dim}, модель говорит {model.total_state_dim}"
                results['errors'].append(msg)
                print(f"    ❌ {msg}")
            else:
                print(f"    ✅ Размерность входа совпадает")

        results['checks']['networks'] = {'status': 'OK'}

    except Exception as e:
        print(f"  ❌ Ошибка проверки сетей: {e}")
        traceback.print_exc()
        results['checks']['networks'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- 4. Проверка памяти ----------
    print("\n[4/8] Проверка памяти модели...")
    try:
        memory_size = len(model.memory)
        print(f"  Обычная память: {memory_size} опытов (max: {model.memory.maxlen})")

        if hasattr(model, 'prioritized_buffer'):
            pb_size = model.prioritized_buffer.size
            pb_max = model.prioritized_buffer.max_size
            print(f"  Приоритетный буфер: {pb_size} опытов (max: {pb_max})")

            if pb_size > 0:
                # Проверка целостности буфера
                priorities = model.prioritized_buffer.priorities[:pb_size]
                if np.isnan(priorities).any():
                    results['errors'].append("Приоритетный буфер содержит NaN в приоритетах")
                    print("  ⚠️ Обнаружены NaN в приоритетах")
                if np.isinf(priorities).any():
                    results['errors'].append("Приоритетный буфер содержит Inf в приоритетах")
                    print("  ⚠️ Обнаружены Inf в приоритетах")
        else:
            print("  Приоритетный буфер: отсутствует")

        # Проверка размера файла памяти
        memory_file = Path(model.memory_config.get('memory_file', 'models/saved_trader/memory_buffer.pkl'))
        if memory_file.exists():
            size = memory_file.stat().st_size
            print(f"  Файл памяти: {memory_file} ({format_size(size)})")
        else:
            print(f"  Файл памяти: не найден")

        results['checks']['memory'] = {
            'status': 'OK',
            'memory_size': memory_size,
            'prioritized_size': pb_size if hasattr(model, 'prioritized_buffer') else 0
        }

    except Exception as e:
        print(f"  ❌ Ошибка проверки памяти: {e}")
        results['checks']['memory'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- 5. Тест forward pass ----------
    print("\n[5/8] Тест forward pass...")
    try:
        model.policy_net.eval()

        # Создаем тестовое состояние
        test_state = torch.randn(1, model.total_state_dim, device=model.device)

        with torch.no_grad():
            start = time.time()
            probs, value, price_pred = model.policy_net(test_state)
            elapsed = time.time() - start

        print(f"  Время выполнения: {elapsed * 1000:.2f} мс")
        print(f"  probs shape: {probs.shape}, sum={probs.sum().item():.3f}")
        print(f"  value: {value.item():.4f}")
        print(f"  price_pred shape: {price_pred.shape}")

        # Проверка выходов
        if torch.isnan(probs).any() or torch.isnan(value).any() or torch.isnan(price_pred).any():
            results['errors'].append("Forward pass вернул NaN")
            print("  ❌ Обнаружены NaN в выходе")
        else:
            print("  ✅ Forward pass успешен")

        results['checks']['forward_pass'] = {
            'status': 'OK',
            'time_ms': elapsed * 1000,
            'probs_sum': probs.sum().item()
        }

    except Exception as e:
        print(f"  ❌ Ошибка forward pass: {e}")
        traceback.print_exc()
        results['checks']['forward_pass'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- 6. Тест выбора стратегии ----------
    print("\n[6/8] Тест выбора стратегии...")
    try:
        # Создаем базовое состояние (без стратегии)
        base_state = torch.randn(model.base_state_dim, device=model.device)

        market_context = {
            'market_sentiment': 0.1,
            'volatility': 0.2,
            'confidence': 0.5,
            'time_of_day': 0.5,
            'ticker_sentiment': 0.0,
            'assigned_horizon': 'day'
        }

        start = time.time()
        action, strategy, confidence = model.choose_action_with_strategy(
            state=base_state,
            ticker='TEST',
            price=100.0,
            market_context=market_context
        )
        elapsed = time.time() - start

        print(f"  Время выполнения: {elapsed * 1000:.2f} мс")
        print(f"  Выбрано действие: {action} ({['BUY', 'HOLD', 'SELL'][action]})")
        print(f"  Стратегия: {strategy}")
        print(f"  Уверенность: {confidence:.3f}")

        results['checks']['choose_action'] = {
            'status': 'OK',
            'time_ms': elapsed * 1000,
            'action': action,
            'strategy': strategy,
            'confidence': confidence
        }

    except Exception as e:
        print(f"  ❌ Ошибка выбора стратегии: {e}")
        traceback.print_exc()
        results['checks']['choose_action'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- 7. Проверка build_state_vector ----------
    print("\n[7/8] Проверка build_state_vector...")
    try:
        test_ticker = "SBER"
        test_price = 300.0
        test_momentum = 0.02
        test_sentiment = 0.1

        # Создаем фиктивные новостные фичи
        news_features = torch.zeros(1, model.news_encoded_dim, device=model.device)

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

        state = model.build_state_vector(
            ticker=test_ticker,
            price=test_price,
            momentum=test_momentum,
            sentiment=test_sentiment,
            news_features=news_features,
            market_data=test_market_data,
            market_sentiment=0.0,
            portfolio=None
        )

        print(f"  Размерность состояния: {state.shape[0]}")
        print(f"  Ожидаемая размерность: {model.base_state_dim}")

        if state.shape[0] != model.base_state_dim:
            msg = f"build_state_vector вернул {state.shape[0]}, ожидалось {model.base_state_dim}"
            results['errors'].append(msg)
            print(f"  ❌ {msg}")
        else:
            print(f"  ✅ Размерность совпадает")

        # Проверка на NaN/Inf
        ok, msg = check_tensor(state, "state_vector")
        if not ok:
            results['errors'].append(msg)
            print(f"  ❌ {msg}")
        else:
            print(f"  ✅ Тензор корректен")

        results['checks']['build_state'] = {
            'status': 'OK',
            'actual_dim': state.shape[0],
            'expected_dim': model.base_state_dim
        }

    except Exception as e:
        print(f"  ❌ Ошибка build_state_vector: {e}")
        traceback.print_exc()
        results['checks']['build_state'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- 8. Проверка сохраненных файлов ----------
    print("\n[8/8] Проверка сохраненных файлов...")
    try:
        saved_files = {}
        model_dir = Path(model.model_dir)

        weights_path = model_dir / 'model_weights.pth'
        if weights_path.exists():
            size = weights_path.stat().st_size
            saved_files['model_weights.pth'] = {'size': format_size(size), 'exists': True}

            # Попытка загрузить веса
            try:
                checkpoint = torch.load(weights_path, map_location='cpu')
                keys = list(checkpoint.keys())
                saved_files['model_weights.pth']['keys'] = keys
                print(f"  model_weights.pth: {format_size(size)}, ключи: {keys}")
            except Exception as e:
                saved_files['model_weights.pth']['load_error'] = str(e)
                print(f"  ⚠️ Ошибка загрузки model_weights.pth: {e}")
        else:
            saved_files['model_weights.pth'] = {'exists': False}
            print("  model_weights.pth: не найден")

        state_path = model_dir / 'model_state.json'
        if state_path.exists():
            size = state_path.stat().st_size
            saved_files['model_state.json'] = {'size': format_size(size), 'exists': True}

            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                saved_files['model_state.json']['keys'] = list(state.keys())
                print(f"  model_state.json: {format_size(size)}, ключи: {list(state.keys())}")
            except Exception as e:
                saved_files['model_state.json']['load_error'] = str(e)
                print(f"  ⚠️ Ошибка загрузки model_state.json: {e}")
        else:
            saved_files['model_state.json'] = {'exists': False}
            print("  model_state.json: не найден")

        results['checks']['saved_files'] = saved_files

    except Exception as e:
        print(f"  ❌ Ошибка проверки файлов: {e}")
        results['checks']['saved_files'] = {'status': 'FAIL', 'error': str(e)}

    # ---------- ИТОГ ----------
    print("\n" + "=" * 80)
    print("ИТОГ ДИАГНОСТИКИ")
    print("=" * 80)

    # Подсчет статусов
    all_ok = True
    for check_name, check_data in results['checks'].items():
        if isinstance(check_data, dict) and check_data.get('status') == 'FAIL':
            all_ok = False
            break

    if results['errors']:
        all_ok = False
        print(f"\n❌ ОБНАРУЖЕНЫ ОШИБКИ ({len(results['errors'])}):")
        for err in results['errors']:
            print(f"  • {err}")
    else:
        print("\n✅ Ошибок не обнаружено")

    if results['warnings']:
        print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ ({len(results['warnings'])}):")
        for warn in results['warnings'][:5]:  # Показываем первые 5
            print(f"  • {warn}")
        if len(results['warnings']) > 5:
            print(f"  ... и еще {len(results['warnings']) - 5}")

    if all_ok and not results['errors']:
        print("\n✅ МОДЕЛЬ ПОЛНОСТЬЮ РАБОТОСПОСОБНА")
        results['overall_status'] = 'HEALTHY'
    else:
        print("\n⚠️ МОДЕЛЬ ИМЕЕТ ПРОБЛЕМЫ, ТРЕБУЕТСЯ ВНИМАНИЕ")
        results['overall_status'] = 'UNHEALTHY'

    # Сохраняем отчет
    output_path = Path('data/model_diagnostics_full.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n📄 Полный отчет сохранен: {output_path}")

    return results


if __name__ == "__main__":
    run_diagnostics()