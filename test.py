#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Диагностика формата price_pred, возвращаемого policy_net
"""

import sys
import os
import torch
import json
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def debug_price_pred_format():
    """Определяем точный формат price_pred от policy_net"""

    print("=" * 60)
    print("ДИАГНОСТИКА ФОРМАТА PRICE_PRED")
    print("=" * 60)

    # 1. Загружаем модель
    try:
        from models.trader_model import AdvancedTraderModel

        print("\n1. Загружаем модель...")
        model = AdvancedTraderModel()
        model.policy_net.eval()
        print(f"   ✅ Модель загружена, device: {model.device}")

    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Создаем тестовые состояния разной размерности
    print("\n2. Создаем тестовые состояния...")

    test_states = {
        "одно состояние (вектор)": torch.randn(156).to(model.device),
        "одно состояние с batch_dim=1": torch.randn(1, 156).to(model.device),
        "батч из 3 состояний": torch.randn(3, 156).to(model.device),
    }

    # 3. Тестируем каждый формат
    print("\n3. Тестируем policy_net с разными входными форматами:")
    print("-" * 60)

    results = {}

    for name, state in test_states.items():
        print(f"\n📌 Тест: {name}")
        print(f"   Входная форма: {state.shape}")

        try:
            with torch.no_grad():
                action_probs, state_value, price_pred = model.policy_net(state)

            print(f"   ✅ Успешно")
            print(f"   action_probs.shape: {action_probs.shape}")
            print(f"   state_value.shape: {state_value.shape}")
            print(f"   price_pred.shape: {price_pred.shape}")
            print(f"   price_pred.dim(): {price_pred.dim()}")
            print(f"   price_pred тип: {price_pred.dtype}")
            print(f"   price_pred значения: {price_pred.cpu().numpy()}")

            # Проверяем softmax с разными dim
            print(f"\n   🔍 Тест softmax:")

            if price_pred.dim() == 1:
                try:
                    softmax_dim0 = torch.softmax(price_pred, dim=0)
                    print(f"      softmax(dim=0): {softmax_dim0.cpu().numpy()}")
                except Exception as e:
                    print(f"      ❌ softmax(dim=0) ошибка: {e}")

                try:
                    softmax_dim1 = torch.softmax(price_pred, dim=1)
                    print(f"      softmax(dim=1): {softmax_dim1.cpu().numpy()}")
                except Exception as e:
                    print(f"      ❌ softmax(dim=1) ошибка: {e}")

            elif price_pred.dim() == 2:
                try:
                    softmax_dim0 = torch.softmax(price_pred, dim=0)
                    print(f"      softmax(dim=0): {softmax_dim0.cpu().numpy()}")
                except Exception as e:
                    print(f"      ❌ softmax(dim=0) ошибка: {e}")

                try:
                    softmax_dim1 = torch.softmax(price_pred, dim=1)
                    print(f"      softmax(dim=1): {softmax_dim1.cpu().numpy()}")
                except Exception as e:
                    print(f"      ❌ softmax(dim=1) ошибка: {e}")

            results[name] = {
                "success": True,
                "input_shape": str(state.shape),
                "price_pred_shape": str(price_pred.shape),
                "price_pred_dim": price_pred.dim(),
                "can_softmax_dim0": price_pred.dim() >= 1,
                "can_softmax_dim1": price_pred.dim() >= 2
            }

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            results[name] = {
                "success": False,
                "error": str(e)
            }

    # 4. Анализируем результаты
    print("\n" + "=" * 60)
    print("4. АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    for name, result in results.items():
        if result.get("success"):
            print(f"\n✅ {name}:")
            print(f"   price_pred.shape: {result['price_pred_shape']}")
            print(f"   price_pred.dim: {result['price_pred_dim']}")
            print(f"   Можно softmax(dim=0): {result['can_softmax_dim0']}")
            print(f"   Можно softmax(dim=1): {result['can_softmax_dim1']}")
        else:
            print(f"\n❌ {name}: {result.get('error')}")

    # 5. Загружаем конфиг для проверки PRICE_PREDICTION_WEIGHT
    print("\n" + "=" * 60)
    print("5. ПРОВЕРКА КОНФИГА")
    print("=" * 60)

    try:
        with open("config/rl_config.json", "r") as f:
            rl_config = json.load(f)
            price_weight = rl_config.get("price_prediction_weight", 0.1)
            print(f"   price_prediction_weight в конфиге: {price_weight}")
    except Exception as e:
        print(f"   Не удалось загрузить rl_config.json: {e}")

    # 6. Проверяем метод predict в модели
    print("\n" + "=" * 60)
    print("6. ПРОВЕРКА МЕТОДА policy_net.forward")
    print("=" * 60)

    try:
        # Смотрим код forward метода
        import inspect
        from models.trader_model import TradingPolicyNetwork

        print("\nСигнатура TradingPolicyNetwork.forward:")
        print(inspect.signature(TradingPolicyNetwork.forward))

        print("\nДокументация:")
        print(inspect.getdoc(TradingPolicyNetwork.forward))

    except Exception as e:
        print(f"   Ошибка инспекции: {e}")

    # 7. Вывод рекомендаций
    print("\n" + "=" * 60)
    print("7. РЕКОМЕНДАЦИИ")
    print("=" * 60)

    # Анализируем результаты для одно состояния (вектор)
    vec_result = results.get("одно состояние (вектор)", {})
    if vec_result.get("success"):
        if vec_result["price_pred_dim"] == 1:
            print("\n✅ Для одного состояния (вектор 156) price_pred имеет dim=1")
            print("   ИСПОЛЬЗУЙТЕ: torch.softmax(price_pred, dim=0)")
        elif vec_result["price_pred_dim"] == 2 and vec_result["price_pred_shape"] == "(1, 3)":
            print("\n✅ Для одного состояния (вектор 156) price_pred имеет форму (1,3)")
            print("   ИСПОЛЬЗУЙТЕ: torch.softmax(price_pred, dim=1)[0]")
        else:
            print(f"\n⚠ Нестандартный формат: {vec_result['price_pred_shape']}")

    # Анализируем результаты для состояния с batch_dim=1
    batch_result = results.get("одно состояние с batch_dim=1", {})
    if batch_result.get("success"):
        print(f"\n✅ Для состояния с batch_dim=1 (форма (1,156)):")
        print(f"   price_pred.shape: {batch_result['price_pred_shape']}")
        if batch_result["price_pred_dim"] == 2:
            print("   ИСПОЛЬЗУЙТЕ: torch.softmax(price_pred, dim=1)[0]")

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ УНИВЕРСАЛЬНАЯ ФУНКЦИЯ:")
    print("=" * 60)
    print("""
def get_pred_probs(price_pred):
    '''Универсальное получение вероятностей из price_pred'''
    if price_pred.dim() == 1:
        # Вектор (3,)
        return torch.softmax(price_pred, dim=0).cpu().numpy()
    elif price_pred.dim() == 2 and price_pred.shape[0] == 1:
        # Матрица (1,3)
        return torch.softmax(price_pred, dim=1).cpu().numpy()[0]
    elif price_pred.dim() == 2:
        # Батч (batch_size, 3)
        return torch.softmax(price_pred, dim=1).cpu().numpy()
    else:
        raise ValueError(f"Неожиданная размерность price_pred: {price_pred.shape}")
    """)


if __name__ == "__main__":
    debug_price_pred_format()