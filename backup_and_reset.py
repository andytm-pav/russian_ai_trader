#!/usr/bin/env python3
# backup_and_reset.py
"""
Скрипт для бэкапа и очистки опыта модели и портфеля
Запуск: python backup_and_reset.py
"""

import os
import sys
import json
import shutil
import pickle
import gzip
from datetime import datetime
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class BackupAndReset:
    """Бэкап и очистка данных модели"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"backups/backup_{self.timestamp}"
        self.dirs_to_backup = [
            "models/saved_trader",
            "data"
        ]
        self.files_to_backup = [
            "config/rl_config.json",
            "config/strategies.json",
            "config/settings.json"
        ]

    def create_backup(self):
        """Создание полного бэкапа"""
        print(f"\n{'=' * 60}")
        print(f"💾 СОЗДАНИЕ БЭКАПА: {self.backup_dir}")
        print(f"{'=' * 60}")

        # Создаем директорию для бэкапа
        os.makedirs(self.backup_dir, exist_ok=True)

        # Бэкап директорий
        for dir_path in self.dirs_to_backup:
            if os.path.exists(dir_path):
                dest = os.path.join(self.backup_dir, dir_path)
                # 🔧 ИСПРАВЛЕНО: создаем родительскую директорию
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                # 🔧 ИСПРАВЛЕНО: используем dirs_exist_ok=True
                shutil.copytree(dir_path, dest, dirs_exist_ok=True)
                print(f"  ✅ Директория {dir_path} -> {dest}")
            else:
                print(f"  ⚠️ Директория {dir_path} не найдена")

        # Бэкап отдельных файлов
        for file_path in self.files_to_backup:
            if os.path.exists(file_path):
                dest = os.path.join(self.backup_dir, file_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(file_path, dest)
                print(f"  ✅ Файл {file_path} -> {dest}")
            else:
                print(f"  ⚠️ Файл {file_path} не найден")

        # Сохраняем информацию о бэкапе
        info = {
            'timestamp': self.timestamp,
            'backup_dir': self.backup_dir,
            'directories': self.dirs_to_backup,
            'files': self.files_to_backup,
            'description': 'Полный бэкап перед очисткой'
        }

        with open(os.path.join(self.backup_dir, 'backup_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\n✅ Бэкап создан: {self.backup_dir}")
        return True

    def reset_model_memory(self):
        """Очистка памяти модели"""
        print(f"\n{'=' * 60}")
        print("🧹 ОЧИСТКА ПАМЯТИ МОДЕЛИ")
        print(f"{'=' * 60}")

        memory_file = "models/saved_trader/memory_buffer.pkl"

        if os.path.exists(memory_file):
            # Переименовываем старый файл (не удаляем)
            backup_memory = f"{memory_file}.backup_{self.timestamp}"
            os.rename(memory_file, backup_memory)
            print(f"  ✅ Старая память сохранена как: {backup_memory}")

        # Создаем пустую память
        empty_memory = []

        # Убеждаемся, что директория существует
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)

        try:
            with gzip.open(memory_file, 'wb') as f:
                pickle.dump(empty_memory, f)
            print(f"  ✅ Создана пустая память: {memory_file} (0 опытов)")
        except:
            with open(memory_file, 'wb') as f:
                pickle.dump(empty_memory, f)
            print(f"  ✅ Создана пустая память: {memory_file} (0 опытов)")

        return True

    def reset_portfolio(self):
        """Сброс портфеля до начального состояния"""
        print(f"\n{'=' * 60}")
        print("💰 СБРОС ПОРТФЕЛЯ")
        print(f"{'=' * 60}")

        portfolio_file = "data/portfolio_state.json"

        if os.path.exists(portfolio_file):
            # Переименовываем старый файл
            backup_portfolio = f"{portfolio_file}.backup_{self.timestamp}"
            os.rename(portfolio_file, backup_portfolio)
            print(f"  ✅ Старый портфель сохранен как: {backup_portfolio}")

        # Загружаем настройки для получения initial_capital
        settings = {}
        settings_file = "config/settings.json"
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

        initial_capital = settings.get('initial_capital_rub', 10000)

        # Убеждаемся, что директория существует
        os.makedirs(os.path.dirname(portfolio_file), exist_ok=True)

        # Создаем пустой портфель
        empty_portfolio = {
            "total_value": initial_capital,
            "cash": initial_capital,
            "positions": {},
            "last_update": datetime.now().isoformat(),
            "initial_capital": initial_capital,
            "reserved_cash": 0,
            "pending_commissions": []
        }

        with open(portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(empty_portfolio, f, indent=2)

        print(f"  ✅ Создан пустой портфель: {portfolio_file}")
        print(f"  ✅ Начальный капитал: {initial_capital}₽")

        return True

    def reset_model_weights(self):
        """Сброс весов модели (опционально)"""
        print(f"\n{'=' * 60}")
        print("🧠 СБРОС ВЕСОВ МОДЕЛИ")
        print(f"{'=' * 60}")

        weights_file = "models/saved_trader/model_weights.pth"
        state_file = "models/saved_trader/model_state.json"

        # Убеждаемся, что директория существует
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)

        if os.path.exists(weights_file):
            backup_weights = f"{weights_file}.backup_{self.timestamp}"
            os.rename(weights_file, backup_weights)
            print(f"  ✅ Старые веса сохранены как: {backup_weights}")

        if os.path.exists(state_file):
            backup_state = f"{state_file}.backup_{self.timestamp}"
            os.rename(state_file, backup_state)
            print(f"  ✅ Старое состояние сохранено как: {backup_state}")

        # Создаем пустой state файл
        empty_state = {
            "error_memory": {},
            "ticker_stats": {},
            "market_sentiment": 0.0,
            "sentiment_history": [],
            "volatility_index": 1.0,
            "strategy_performance": {},
            "strategy_memory": [],
            "strategies": {},
            "save_time": datetime.now().isoformat()
        }

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(empty_state, f, indent=2)

        print(f"  ✅ Создано пустое состояние модели")
        print(f"  ⚠️ Веса модели будут созданы заново при первом запуске")

        return True

    def list_backups(self):
        """Показать все доступные бэкапы"""
        print(f"\n{'=' * 60}")
        print("📋 ДОСТУПНЫЕ БЭКАПЫ")
        print(f"{'=' * 60}")

        backup_root = "backups"
        if not os.path.exists(backup_root):
            print("  ❌ Нет бэкапов")
            return []

        backups = []
        for item in sorted(os.listdir(backup_root)):
            if item.startswith("backup_"):
                backup_path = os.path.join(backup_root, item)
                info_file = os.path.join(backup_path, 'backup_info.json')

                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    backups.append({
                        'dir': backup_path,
                        'timestamp': info.get('timestamp', item),
                        'date': item.replace('backup_', '')
                    })
                else:
                    backups.append({
                        'dir': backup_path,
                        'timestamp': item,
                        'date': item.replace('backup_', '')
                    })

                # Подсчет размера
                total_size = 0
                for file_path in Path(backup_path).rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                print(f"  📁 {item} ({total_size / 1024:.1f} KB)")

        return backups

    def restore_backup(self, backup_name=None):
        """Восстановление из бэкапа"""
        print(f"\n{'=' * 60}")
        print("🔄 ВОССТАНОВЛЕНИЕ ИЗ БЭКАПА")
        print(f"{'=' * 60}")

        backups = self.list_backups()
        if not backups:
            print("  ❌ Нет бэкапов для восстановления")
            return False

        if backup_name is None:
            print("\n  Выберите бэкап для восстановления:")
            for i, b in enumerate(backups):
                print(f"  {i + 1}. {b['dir']}")
            print(f"  {len(backups) + 1}. Отмена")

            try:
                choice = int(input("\n  Ваш выбор: ")) - 1
                if choice < 0 or choice >= len(backups):
                    print("  ❌ Отменено")
                    return False
                backup_dir = backups[choice]['dir']
            except:
                print("  ❌ Отменено")
                return False
        else:
            backup_dir = os.path.join("backups", backup_name)
            if not os.path.exists(backup_dir):
                print(f"  ❌ Бэкап {backup_name} не найден")
                return False

        print(f"\n  🔄 Восстановление из: {backup_dir}")

        # Создаем бэкап текущего состояния перед восстановлением
        current_backup = BackupAndReset()
        current_backup.create_backup()

        # Восстанавливаем директории
        for dir_path in self.dirs_to_backup:
            src = os.path.join(backup_dir, dir_path)
            if os.path.exists(src):
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                shutil.copytree(src, dir_path, dirs_exist_ok=True)
                print(f"  ✅ Восстановлена директория {dir_path}")

        # Восстанавливаем файлы
        for file_path in self.files_to_backup:
            src = os.path.join(backup_dir, file_path)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copy2(src, file_path)
                print(f"  ✅ Восстановлен файл {file_path}")

        print(f"\n✅ Восстановление завершено!")
        return True


def main():
    """Основная функция"""

    print("=" * 60)
    print("🔧 ИНСТРУМЕНТ БЭКАПА И ОЧИСТКИ")
    print("=" * 60)

    backup_tool = BackupAndReset()

    while True:
        print("\nДоступные действия:")
        print("  1. 📦 Создать полный бэкап")
        print("  2. 🧹 Очистить память модели")
        print("  3. 💰 Сбросить портфель")
        print("  4. 🧠 Сбросить веса модели")
        print("  5. 📋 Список бэкапов")
        print("  6. 🔄 Восстановить из бэкапа")
        print("  7. 🚪 Выход")
        print("  8. ⚡ ПОЛНЫЙ СБРОС (всё сразу)")

        choice = input("\nВаш выбор (1-8): ").strip()

        if choice == '1':
            backup_tool.create_backup()

        elif choice == '2':
            confirm = input("Вы уверены, что хотите очистить память модели? (yes/no): ")
            if confirm.lower() == 'yes':
                backup_tool.create_backup()  # Сначала бэкап
                backup_tool.reset_model_memory()
            else:
                print("❌ Отменено")

        elif choice == '3':
            confirm = input("Вы уверены, что хотите сбросить портфель? (yes/no): ")
            if confirm.lower() == 'yes':
                backup_tool.create_backup()  # Сначала бэкап
                backup_tool.reset_portfolio()
            else:
                print("❌ Отменено")

        elif choice == '4':
            confirm = input("Вы уверены, что хотите сбросить веса модели? (yes/no): ")
            if confirm.lower() == 'yes':
                backup_tool.create_backup()  # Сначала бэкап
                backup_tool.reset_model_weights()
            else:
                print("❌ Отменено")

        elif choice == '5':
            backup_tool.list_backups()

        elif choice == '6':
            backup_tool.restore_backup()

        elif choice == '7':
            print("\n👋 Выход")
            break

        elif choice == '8':
            print("\n⚠️  ПОЛНЫЙ СБРОС ВСЕХ ДАННЫХ ⚠️")
            print("Будут очищены:")
            print("  • Память модели")
            print("  • Портфель")
            print("  • Веса модели")
            print("  • Состояние модели")

            confirm = input("\nВы абсолютно уверены? (введите 'ПОЛНЫЙ СБРОС'): ")
            if confirm == "ПОЛНЫЙ СБРОС":
                print("\n📦 Создаю бэкап перед сбросом...")
                backup_tool.create_backup()
                backup_tool.reset_model_memory()
                backup_tool.reset_portfolio()
                backup_tool.reset_model_weights()
                print("\n✅ Полный сброс завершен!")
            else:
                print("❌ Отменено")

        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    main()