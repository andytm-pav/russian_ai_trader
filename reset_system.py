#!/usr/bin/env python3
"""
СБРОС СИСТЕМЫ К НАЧАЛЬНОМУ СОСТОЯНИЮ
Очищает portfolio_state.json, price_history.json, сбрасывает дневную статистику
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')


def reset_system(confirm: bool = True):
    """Сброс системы к дефолтным значениям"""

    print("\n" + "=" * 70)
    print("🔄 СБРОС СИСТЕМЫ К НАЧАЛЬНОМУ СОСТОЯНИЮ")
    print("=" * 70)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Загружаем настройки для получения initial_capital
    initial_capital = 10000.0
    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
            initial_capital = settings.get('initial_capital_rub', 10000.0)
    except Exception as e:
        print(f"⚠️ Не удалось загрузить settings.json: {e}")
        print(f"   Использую initial_capital = 10000.0₽")

    results = {}
    files_cleaned = []
    files_created = []
    files_error = []

    # ===== 1. СБРОС PORTFOLIO_STATE.JSON =====
    portfolio_file = Path('data/portfolio_state.json')
    print(f"\n📁 Портфель: {portfolio_file}")

    if portfolio_file.exists():
        # Создаём бэкап
        backup_file = portfolio_file.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        try:
            import shutil
            shutil.copy2(portfolio_file, backup_file)
            print(f"   ✅ Бэкап сохранён: {backup_file}")
        except Exception as e:
            print(f"   ⚠️ Не удалось создать бэкап: {e}")

        # Читаем старые данные для информации
        try:
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                old_state = json.load(f)
            old_cash = old_state.get('cash', 0)
            old_positions = len(old_state.get('positions', {}))
            old_total = old_state.get('total_value', 0)
            old_reserved = old_state.get('reserved_cash', 0)
            print(f"   Старое состояние: cash={old_cash:,.0f}₽, positions={old_positions}, "
                  f"total_value={old_total:,.0f}₽, reserved={old_reserved:,.0f}₽")
        except:
            pass

    # Создаём новое состояние
    new_state = {
        'total_value': initial_capital,
        'cash': initial_capital,
        'reserved_cash': 0.0,
        'positions': {},
        'sector_allocation': {},
        'correlation_matrix': {},
        'trade_history': [],
        'strategy_positions': {},
        'initial_capital': initial_capital,
        'total_commission': 0.0,
        'total_trades': 0,
        'total_pnl': 0.0,
        'commission_spent_today': 0.0,
        'daily_trades': [],
        'pending_commissions': [],
        'last_update': datetime.now().isoformat(),
        'stats': {
            'cash': initial_capital,
            'positions_count': 0,
            'total_trades': 0,
            'initial_capital': initial_capital,
            'current_capital': initial_capital,
            'strategies_count': 0
        }
    }

    try:
        with open(portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(new_state, f, indent=2, default=str)
        print(f"   ✅ Сброшен: cash={initial_capital:,.0f}₽, positions=0, reserved=0")
        files_cleaned.append(str(portfolio_file))
    except Exception as e:
        print(f"   ❌ Ошибка записи: {e}")
        files_error.append(str(portfolio_file))

    # ===== 2. ОЧИСТКА PRICE_HISTORY.JSON =====
    price_history_file = Path('data/price_history.json')
    print(f"\n📁 История цен: {price_history_file}")

    if price_history_file.exists():
        # Создаём бэкап
        backup_file = price_history_file.with_suffix(f'.json.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        try:
            import shutil
            shutil.copy2(price_history_file, backup_file)
            print(f"   ✅ Бэкап сохранён: {backup_file}")
        except Exception as e:
            print(f"   ⚠️ Не удалось создать бэкап: {e}")

    # Создаём пустой файл
    try:
        with open(price_history_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        print(f"   ✅ Очищен (пустой словарь)")
        files_cleaned.append(str(price_history_file))
    except Exception as e:
        print(f"   ❌ Ошибка записи: {e}")
        files_error.append(str(price_history_file))

    # ===== 3. ОЧИСТКА ПАМЯТИ МОДЕЛИ (ОПЦИОНАЛЬНО) =====
    memory_file = Path('models/saved_trader/memory_buffer.pkl')
    print(f"\n📁 Память модели: {memory_file}")

    if memory_file.exists():
        backup_file = memory_file.with_suffix(f'.pkl.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        try:
            import shutil
            shutil.copy2(memory_file, backup_file)
            print(f"   ✅ Бэкап сохранён: {backup_file}")
        except Exception as e:
            print(f"   ⚠️ Не удалось создать бэкап: {e}")

        try:
            memory_file.unlink()
            print(f"   ✅ Удалён (модель начнёт с чистой памяти)")
            files_cleaned.append(str(memory_file))
        except Exception as e:
            print(f"   ❌ Ошибка удаления: {e}")
            files_error.append(str(memory_file))
    else:
        print(f"   ⚠️ Файл не найден (уже очищен или не создавался)")

    # ===== 4. СБРОС ДНЕВНЫХ ОТЧЁТОВ =====
    backoffice_dir = Path('data/backoffice')
    print(f"\n📁 Бэкофис: {backoffice_dir}")

    if backoffice_dir.exists():
        count = 0
        for f in backoffice_dir.glob('*.json'):
            try:
                f.unlink()
                count += 1
            except:
                pass
        if count > 0:
            print(f"   ✅ Удалено {count} файлов отчётов")
            files_cleaned.append(f"{backoffice_dir} ({count} файлов)")
        else:
            print(f"   ⚠️ Нет файлов для очистки")
    else:
        print(f"   ⚠️ Директория не найдена")

    # ===== 5. СБРОС ДНЕВНЫХ ОТЧЁТОВ В КОРНЕ DATA =====
    data_dir = Path('data')
    print(f"\n📁 Дневные отчёты: {data_dir}")

    count = 0
    for f in data_dir.glob('daily_report_*.json'):
        try:
            f.unlink()
            count += 1
        except:
            pass
    if count > 0:
        print(f"   ✅ Удалено {count} дневных отчётов")
        files_cleaned.append(f"daily_report_*.json ({count} файлов)")
    else:
        print(f"   ⚠️ Нет файлов для очистки")

    # ===== 6. СБРОС TRAINING EXPORT =====
    training_export = Path('data/training_export.json')
    if training_export.exists():
        try:
            training_export.unlink()
            print(f"\n📁 Training export: удалён")
            files_cleaned.append(str(training_export))
        except:
            pass

    # ===== ИТОГ =====
    print("\n" + "=" * 70)
    print("📋 ИТОГ СБРОСА")
    print("=" * 70)

    print(f"\n   Начальный капитал: {initial_capital:,.0f}₽")
    print(f"\n   Очищено файлов: {len(files_cleaned)}")
    for f in files_cleaned:
        print(f"   ✅ {f}")

    if files_error:
        print(f"\n   Ошибок: {len(files_error)}")
        for f in files_error:
            print(f"   ❌ {f}")

    print(f"\n   Бэкапы сохранены с суффиксом .backup_YYYYMMDD_HHMMSS")

    if not files_error:
        print(f"\n   ✅ СИСТЕМА СБРОШЕНА К НАЧАЛЬНОМУ СОСТОЯНИЮ!")
        print(f"   Можно запускать: python main.py")
    else:
        print(f"\n   ⚠️ СБРОС ВЫПОЛНЕН С ОШИБКАМИ. Проверьте права доступа.")

    print("\n" + "=" * 70)
    print("✅ СБРОС ЗАВЕРШЁН")
    print("=" * 70)

    return len(files_error) == 0


if __name__ == "__main__":
    # Запрос подтверждения
    print("\n⚠️ ВНИМАНИЕ!")
    print("Этот скрипт сбросит систему к начальному состоянию:")
    print("  — Портфель: 10 000₽, 0 позиций")
    print("  — История цен: очищена")
    print("  — Память модели: удалена")
    print("  — Дневные отчёты: удалены")
    print("  — Бэкапы будут созданы автоматически")

    response = input("\nПродолжить? (yes/no): ").strip().lower()

    if response in ['yes', 'y', 'да', 'д']:
        reset_system()
    else:
        print("❌ Сброс отменён.")