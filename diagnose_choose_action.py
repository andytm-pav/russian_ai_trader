#!/usr/bin/env python3
# diagnose_choose_action_full.py
"""
Полноценная диагностика choose_action_with_strategy
Запуск: python diagnose_choose_action_full.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.trader_model import trader_model_instance
    from models.smart_broker import SmartPortfolioBroker

    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Не удалось загрузить модели: {e}")
    MODEL_LOADED = False
    trader_model_instance = None


class FullChooseActionDiagnostic:
    """Полноценная диагностика метода choose_action_with_strategy"""

    def __init__(self):
        self.model = None
        self.broker = None
        self.issues = []
        self.test_results = {}

    def load_model(self):
        """Загрузка модели со всем контекстом"""
        if not MODEL_LOADED or not trader_model_instance:
            print("❌ Модель не загружена")
            return False

        self.model = trader_model_instance
        print(f"✅ Модель загружена")
        print(f"  • policy_net.state_dim: {self.model.policy_net.state_dim}")
        print(f"  • Стратегий: {len(self.model.strategies)}")
        print(f"  • Устройство: {self.model.device}")
        print(f"  • exploration_rate: {self.model.exploration_rate}")
        print(f"  • confidence_boost_factor: {self.model.confidence_boost_factor}")

        # Проверяем наличие стратегий
        if not self.model.strategies:
            self.issues.append("❌ Нет стратегий в модели")
            return False

        return True

    def create_test_context(self):
        """Создание полноценного market_context"""
        return {
            'current_strategy': 'balanced',
            'market_sentiment': 0.1,
            'volatility': 0.2,
            'hour': 14,
            'day': 3
        }

    def test_with_real_state(self, dim: int) -> bool:
        """Тестирование с реальным состоянием, как в production"""
        print(f"\n  🔬 ТЕСТ С РАЗМЕРНОСТЬЮ {dim}:")

        try:
            # 1. Создаем тестовое состояние
            test_state = torch.randn(dim, device=self.model.device)

            # 2. Создаем контекст
            market_context = self.create_test_context()

            # 3. Вызываем метод
            action, strategy, confidence = self.model.choose_action_with_strategy(
                state=test_state,
                ticker="TEST_TICKER",
                price=100.0,
                market_context=market_context
            )

            # 4. Проверяем результат
            print(f"    ✓ Результат: action={action}, strategy={strategy}, confidence={confidence:.3f}")

            # 5. Проверяем, что стратегия существует
            if strategy in self.model.strategies:
                print(f"    ✓ Стратегия '{strategy}' найдена в конфиге")
            else:
                self.issues.append(f"❌ Стратегия '{strategy}' не найдена в конфиге")
                print(f"    ❌ Стратегия '{strategy}' не найдена в конфиге")

            # 6. Проверяем, что action в допустимом диапазоне (0-2)
            if 0 <= action <= 2:
                print(f"    ✓ Action {action} в допустимом диапазоне")
            else:
                self.issues.append(f"❌ Action {action} вне диапазона 0-2")

            # Сохраняем результат
            self.test_results[dim] = {
                'success': True,
                'action': action,
                'strategy': strategy,
                'confidence': confidence
            }

            return True

        except Exception as e:
            print(f"    ❌ ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
            self.issues.append(f"Ошибка при размерности {dim}: {e}")
            self.test_results[dim] = {'success': False, 'error': str(e)}
            return False

    def test_create_strategy_state(self, dim: int) -> bool:
        """Тестирование _create_strategy_state с реальными параметрами"""
        print(f"\n  🔬 ТЕСТ _create_strategy_state С РАЗМЕРНОСТЬЮ {dim}:")

        try:
            # Берем первую стратегию для теста
            strategy_name = list(self.model.strategies.keys())[0]
            strategy_params = self.model.strategies[strategy_name]

            # Создаем тестовое состояние
            test_state = torch.randn(dim, device=self.model.device)

            print(f"    Вход: {test_state.shape}")
            print(f"    Стратегия: {strategy_name}")

            # Вызываем метод
            result_state = self.model._create_strategy_state(test_state, strategy_params)

            print(f"    Выход: {result_state.shape}")
            print(f"    Ожидалось: {self.model.policy_net.state_dim}")

            # Проверяем результат
            if result_state.shape[-1] == self.model.policy_net.state_dim:
                print(f"    ✓ Размерность корректна")
                return True
            else:
                self.issues.append(
                    f"❌ Неверная выходная размерность: {result_state.shape[-1]}, ожидалось {self.model.policy_net.state_dim}")
                return False

        except Exception as e:
            print(f"    ❌ ОШИБКА: {e}")
            self.issues.append(f"Ошибка в _create_strategy_state для {dim}: {e}")
            return False

    def test_all_dimensions(self):
        """Тестирование всех размерностей"""
        print("\n" + "=" * 60)
        print("🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ ВСЕХ РАЗМЕРНОСТЕЙ")
        print("=" * 60)

        test_dims = [150, 151, 156, 157]

        for dim in test_dims:
            print(f"\n{'-' * 40}")
            success = self.test_with_real_state(dim)

            # Дополнительно тестируем _create_strategy_state для нецелевых размерностей
            if dim != self.model.policy_net.state_dim:
                self.test_create_strategy_state(dim)

    def test_with_batch(self):
        """Тестирование с батчем данных"""
        print("\n" + "=" * 60)
        print("🧪 ТЕСТИРОВАНИЕ С БАТЧЕМ ДАННЫХ")
        print("=" * 60)

        try:
            batch_size = 4
            dim = 151

            # Создаем батч состояний
            batch_states = torch.randn(batch_size, dim, device=self.model.device)

            print(f"\n  Батч состояний: {batch_states.shape}")

            # Создаем стратегические состояния для всего батча
            strategy_name = list(self.model.strategies.keys())[0]
            strategy_params = self.model.strategies[strategy_name]

            # Применяем _create_strategy_state ко всему батчу
            batch_strategy_states = self.model._create_strategy_state(batch_states, strategy_params)
            print(f"  После _create_strategy_state: {batch_strategy_states.shape}")

            # Проверяем, что policy_net принимает батч
            with torch.no_grad():
                action_probs, state_values, price_pred = self.model.policy_net(batch_strategy_states)
                print(f"  action_probs: {action_probs.shape}")
                print(f"  state_values: {state_values.shape}")
                print(f"  price_pred: {price_pred.shape}")

            if action_probs.shape[0] == batch_size:
                print(f"  ✓ Батч обработан корректно")
            else:
                self.issues.append("❌ Проблема с обработкой батча")

        except Exception as e:
            print(f"  ❌ Ошибка при тестировании батча: {e}")
            self.issues.append(f"Ошибка с батчем: {e}")

    def test_with_real_portfolio(self):
        """Тестирование с реальным портфелем"""
        print("\n" + "=" * 60)
        print("🧪 ТЕСТИРОВАНИЕ С РЕАЛЬНЫМ ПОРТФЕЛЕМ")
        print("=" * 60)

        try:
            # Загружаем брокера
            settings_path = "config/settings.json"
            if os.path.exists(settings_path):
                import json
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                self.broker = SmartPortfolioBroker(settings)
                print(f"  ✅ Брокер загружен")

                # Даем модели доступ к портфелю
                self.model.portfolio = self.broker.portfolio

                # Проверяем build_state_vector с реальными данными
                if hasattr(self.model, 'build_state_vector'):
                    test_state = self.model.build_state_vector(
                        ticker="SBER",
                        price=250.0,
                        momentum=0.1,
                        sentiment=0.2,
                        news_features=torch.zeros(1, 128, device=self.model.device),
                        market_data={'volume': 1000000, 'volatility': 0.15},
                        market_sentiment=0.1
                    )
                    print(f"  build_state_vector: {test_state.shape}")

                    if test_state.shape[-1] == 151:
                        print(f"  ✓ build_state_vector работает корректно")
                    else:
                        self.issues.append(f"❌ build_state_vector вернул {test_state.shape[-1]}, ожидалось 151")

            else:
                print(f"  ⚠️ Файл настроек не найден, пропускаем")

        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            self.issues.append(f"Ошибка при тестировании с портфелем: {e}")

    def generate_report(self):
        """Генерация отчета"""
        print("\n" + "=" * 60)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 60)

        print(f"\n✅ Успешные тесты:")
        for dim, result in self.test_results.items():
            if result.get('success'):
                print(f"  • {dim}: action={result.get('action')}, strategy={result.get('strategy')}")

        if self.issues:
            print(f"\n❌ Найденные проблемы:")
            for issue in self.issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ ПРОБЛЕМ НЕ НАЙДЕНО!")

        print(f"\n📈 Статистика:")
        print(f"  • Всего тестов: {len(self.test_results)}")
        print(f"  • Успешно: {sum(1 for r in self.test_results.values() if r.get('success'))}")
        print(f"  • Провалено: {sum(1 for r in self.test_results.values() if not r.get('success'))}")


def main():
    """Основная функция"""

    print("=" * 60)
    print("🔍 ПОЛНОЦЕННАЯ ДИАГНОСТИКА choose_action_with_strategy")
    print("=" * 60)

    diag = FullChooseActionDiagnostic()

    if not diag.load_model():
        print("❌ Не удалось загрузить модель")
        return

    # Запускаем все тесты
    diag.test_all_dimensions()
    diag.test_with_batch()
    diag.test_with_real_portfolio()

    # Генерируем отчет
    diag.generate_report()

    if diag.issues:
        print("\n🔧 Найдены проблемы, требующие исправления")
    else:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")


if __name__ == "__main__":
    main()