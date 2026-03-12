#!/usr/bin/env python3
# update_model_strategies.py
"""
Скрипт для добавления новых стратегий из config/strategies.json в model_state.json
Сохраняет весь существующий опыт и статистику
"""

import json
import os
from datetime import datetime
from collections import defaultdict


class StrategyUpdater:
    """Обновление стратегий в model_state.json"""

    def __init__(self):
        # ✅ ПРАВИЛЬНЫЕ ПУТИ согласно структуре проекта
        self.strategies_file = "config/strategies.json"  # ← strategies.json, не settings.json!
        self.model_state_file = "models/saved_trader/model_state.json"
        self.backup_file = f"models/saved_trader/model_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def load_json(self, filepath):
        """Загрузка JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки {filepath}: {e}")
            return None

    def save_json(self, filepath, data):
        """Сохранение JSON файла"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Сохранено: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения {filepath}: {e}")
            return False

    def create_backup(self):
        """Создание бэкапа model_state.json"""
        if os.path.exists(self.model_state_file):
            import shutil
            shutil.copy2(self.model_state_file, self.backup_file)
            print(f"✅ Бэкап создан: {self.backup_file}")
            return True
        return False

    def find_new_strategies(self, strategies_config, model_strategies):
        """Поиск новых стратегий, которых нет в model_state"""
        new_strategies = {}
        for name, config in strategies_config.items():
            if name not in model_strategies:
                new_strategies[name] = config
                print(f"  📌 Найдена новая стратегия: {name}")
        return new_strategies

    def update_model_state(self):
        """Обновление model_state.json с новыми стратегиями"""

        print("=" * 60)
        print("🔄 ОБНОВЛЕНИЕ СТРАТЕГИЙ В model_state.json")
        print("=" * 60)

        # 1. Загружаем конфиги
        strategies_config = self.load_json(self.strategies_file)
        if not strategies_config:
            print(f"❌ Не удалось загрузить {self.strategies_file}")
            return False

        model_state = self.load_json(self.model_state_file)
        if not model_state:
            print(f"❌ Не удалось загрузить {self.model_state_file}")
            return False

        # 2. Создаем бэкап
        self.create_backup()

        # 3. Получаем стратегии из обоих файлов
        config_strategies = strategies_config.get('strategies', {})
        model_strategies = model_state.get('strategies', {})

        print(f"\n📊 Текущее состояние:")
        print(f"  • Стратегий в strategies.json: {len(config_strategies)}")
        print(f"  • Стратегий в model_state.json: {len(model_strategies)}")

        # 4. Находим новые стратегии
        new_strategies = self.find_new_strategies(config_strategies, model_strategies)

        if not new_strategies:
            print("\n✅ Новых стратегий не найдено. Все стратегии уже есть в model_state.")
            return True

        print(f"\n🔍 Найдено {len(new_strategies)} новых стратегий:")
        for name in new_strategies.keys():
            print(f"  • {name}")

        # 5. Добавляем новые стратегии в model_state
        for name, config in new_strategies.items():
            model_strategies[name] = config

            # Добавляем запись в strategy_performance для новой стратегии
            if 'strategy_performance' not in model_state:
                model_state['strategy_performance'] = {}

            model_state['strategy_performance'][name] = {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'win_rate': 0.5
            }

            print(f"  ✅ Добавлена стратегия: {name}")

        # 6. Обновляем strategies в model_state
        model_state['strategies'] = model_strategies

        # 7. Добавляем метку об обновлении
        model_state['strategies_updated'] = datetime.now().isoformat()

        # 8. Сохраняем обновленный model_state
        if self.save_json(self.model_state_file, model_state):
            print(f"\n✅ model_state.json успешно обновлен!")
            print(f"  • Теперь стратегий в model_state: {len(model_strategies)}")
            print(f"  • Добавлены: {', '.join(new_strategies.keys())}")

            # Показываем различия в параметрах
            print(f"\n📋 Параметры новых стратегий:")
            for name, config in new_strategies.items():
                print(f"\n  {name}:")
                print(f"    • target_hold_time: {config.get('target_hold_time_hours', '?')}ч")
                print(f"    • risk_multiplier: {config.get('risk_multiplier', '?')}")
                print(f"    • stop_loss: {config.get('stop_loss_percent', '?')}%")
                print(f"    • take_profit: {config.get('take_profit_percent', '?')}%")

            return True
        else:
            print(f"\n❌ Ошибка при сохранении")
            return False

    def verify_update(self):
        """Проверка, что обновление прошло успешно"""
        model_state = self.load_json(self.model_state_file)
        strategies_config = self.load_json(self.strategies_file)

        if not model_state or not strategies_config:
            return

        print("\n" + "=" * 60)
        print("🔍 ПРОВЕРКА ОБНОВЛЕНИЯ")
        print("=" * 60)

        config_strategies = strategies_config.get('strategies', {})
        model_strategies = model_state.get('strategies', {})

        missing_in_model = []
        for name in config_strategies.keys():
            if name not in model_strategies:
                missing_in_model.append(name)

        if missing_in_model:
            print(f"❌ В model_state все еще отсутствуют: {missing_in_model}")
        else:
            print(f"✅ Все стратегии из strategies.json присутствуют в model_state.json")
            print(f"  • Всего стратегий: {len(model_strategies)}")

        # Проверка strategy_performance
        perf = model_state.get('strategy_performance', {})
        missing_perf = []
        for name in model_strategies.keys():
            if name not in perf:
                missing_perf.append(name)

        if missing_perf:
            print(f"⚠️ Для стратегий нет записи в strategy_performance: {missing_perf}")
        else:
            print(f"✅ Для всех стратегий есть записи в strategy_performance")


def main():
    """Основная функция"""

    updater = StrategyUpdater()

    print("\nДоступные действия:")
    print("  1. 🔄 Обновить model_state.json (добавить новые стратегии)")
    print("  2. 🔍 Проверить соответствие стратегий")
    print("  3. 🚪 Выход")

    choice = input("\nВаш выбор (1-3): ").strip()

    if choice == '1':
        if updater.update_model_state():
            updater.verify_update()
    elif choice == '2':
        updater.verify_update()
    elif choice == '3':
        print("👋 Выход")
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    main()