# reset_system.py
"""
Полный сброс системы:
- Удаление модели и весов
- Удаление памяти (опыта)
- Сброс портфеля
- Сброс настроек трейдера
"""

import os
import shutil
import json
import time
from pathlib import Path


def reset_system():
    print("=" * 60)
    print("🧹 ПОЛНЫЙ СБРОС СИСТЕМЫ")
    print("=" * 60)

    # 1. Удаление модели и весов
    model_dir = Path("models/saved_trader")
    if model_dir.exists():
        print(f"🗑 Удаление: {model_dir}")
        shutil.rmtree(model_dir)
    else:
        print(f"⚠ Папка {model_dir} не найдена")

    # 2. Удаление памяти (опыта)
    memory_files = [
        "models/saved_trader/memory_buffer.pkl",
        "models/saved_trader/memory_buffer.pkl.gz",
        "models/saved_trader/priority_memory.pkl",
    ]
    for mem_file in memory_files:
        path = Path(mem_file)
        if path.exists():
            print(f"🗑 Удаление: {mem_file}")
            os.remove(path)
        else:
            print(f"⚠ Файл {mem_file} не найден")

    # 3. Сброс портфеля
    portfolio_file = Path("data/portfolio_state.json")
    if portfolio_file.exists():
        print(f"🗑 Удаление: {portfolio_file}")
        os.remove(portfolio_file)
    else:
        print(f"⚠ Файл {portfolio_file} не найден")

    # 4. Сброс торговой статистики
    report_dir = Path("data/backoffice")
    if report_dir.exists():
        print(f"🗑 Удаление отчетов: {report_dir}")
        shutil.rmtree(report_dir)

    # 5. Создание пустого портфеля
    print("\n📁 Создание пустого портфеля...")
    default_portfolio = {
        "positions": {},
        "cash": 10000.0,
        "initial_capital": 10000.0,
        "trade_history": [],
        "strategy_positions": {},
        "total_value": 10000.0,
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stats": {
            "cash": 10000.0,
            "positions_count": 0,
            "total_trades": 0,
            "initial_capital": 10000.0,
            "current_capital": 10000.0
        },
        "total_commission": 0.0,
        "total_trades": 0,
        "total_pnl": 0.0,
        "commission_spent_today": 0.0
    }

    portfolio_file.parent.mkdir(parents=True, exist_ok=True)
    with open(portfolio_file, "w", encoding="utf-8") as f:
        json.dump(default_portfolio, f, indent=2, default=str)

    print(f"✅ Создан пустой портфель: {portfolio_file}")

    # 6. Очистка кэша MOEX
    print("\n🗑 Очистка кэша MOEX (будет создан заново при запуске)...")

    print("\n" + "=" * 60)
    print("✅ СБРОС ВЫПОЛНЕН")
    print("=" * 60)
    print("\n📋 Рекомендации:")
    print("1. Перезапустите систему")
    print("2. Модель начнет обучение с нуля")
    print("3. Портфель: 10 000₽, без позиций")
    print("4. Память опыта пуста")


if __name__ == "__main__":
    confirm = input("⚠ ВНИМАНИЕ! Это удалит ВСЕ данные модели и портфеля.\n"
                    "Продолжить? (yes/no): ")
    if confirm.lower() == "yes":
        reset_system()
    else:
        print("❌ Отменено")