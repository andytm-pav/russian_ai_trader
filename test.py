#!/usr/bin/env python3
# reset_everything.py
"""
Полный сброс: удаление памяти, модели, портфеля
СОЗДАЕТ БЭКАП ПЕРЕД УДАЛЕНИЕМ!
"""

import os
import shutil
import json
from datetime import datetime


def create_backup():
    """Создание бэкапа перед удалением"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"backups/full_reset_{timestamp}"

    print(f"\n📦 СОЗДАНИЕ БЭКАПА: {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)

    # Что бэкапим
    dirs_to_backup = ['models/saved_trader', 'data']

    for dir_path in dirs_to_backup:
        if os.path.exists(dir_path):
            dest = os.path.join(backup_dir, dir_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copytree(dir_path, dest)
            print(f"  ✅ {dir_path}")

    print(f"✅ Бэкап создан: {backup_dir}")
    return backup_dir


def delete_memory():
    """Удаление файла памяти"""
    memory_file = 'models/saved_trader/memory_buffer.pkl'
    if os.path.exists(memory_file):
        os.remove(memory_file)
        print(f"  ✅ Удален: {memory_file}")


def delete_model():
    """Удаление весов и состояния модели"""
    files_to_delete = [
        'models/saved_trader/model_weights.pth',
        'models/saved_trader/model_state.json'
    ]
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  ✅ Удален: {file_path}")


def reset_portfolio():
    """Сброс портфеля до начального состояния"""
    portfolio_file = 'data/portfolio_state.json'

    # Создаем пустой портфель
    empty_portfolio = {
        "total_value": 10000,
        "cash": 10000,
        "positions": {},
        "last_update": datetime.now().isoformat(),
        "initial_capital": 10000,
        "reserved_cash": 0,
        "pending_commissions": []
    }

    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(empty_portfolio, f, indent=2)
    print(f"  ✅ Создан пустой портфель: {portfolio_file}")


def main():
    print("=" * 60)
    print("🧹 ПОЛНЫЙ СБРОС МОДЕЛИ И ПАМЯТИ")
    print("=" * 60)
    print("\n⚠️  БУДУТ УДАЛЕНЫ:")
    print("  • memory_buffer.pkl (вся память)")
    print("  • model_weights.pth (веса модели)")
    print("  • model_state.json (состояние модели)")
    print("  • portfolio_state.json (портфель)")

    confirm = input("\nВведите 'ПОЛНЫЙ СБРОС' для подтверждения: ")

    if confirm == "ПОЛНЫЙ СБРОС":
        backup_dir = create_backup()
        print("\n🔥 УДАЛЕНИЕ:")
        delete_memory()
        delete_model()
        reset_portfolio()
        print("\n✅ ПОЛНЫЙ СБРОС ЗАВЕРШЕН!")
        print(f"📦 Бэкап сохранен в: {backup_dir}")
        print("\n🚀 Теперь можно запускать систему с чистого листа")
    else:
        print("❌ Отменено")


if __name__ == "__main__":
    main()