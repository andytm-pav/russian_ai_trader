#!/usr/bin/env python3
"""
СБРОС СИСТЕМЫ К НАЧАЛЬНОМУ СОСТОЯНИЮ (v2)
Очищает: portfolio_state.json, price_history.json, model_weights.pth, memory_buffer.pkl
"""

import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')


def reset_system():
    """Сброс системы к дефолтным значениям"""

    print("\n" + "=" * 70)
    print("🔄 СБРОС СИСТЕМЫ К НАЧАЛЬНОМУ СОСТОЯНИЮ (v2)")
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

    files_cleaned = []
    files_created = []
    files_error = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===== 1. СБРОС PORTFOLIO_STATE.JSON =====
    portfolio_file = Path('data/portfolio_state.json')
    print(f"\n📁 Портфель: {portfolio_file}")

    if portfolio_file.exists():
        backup_file = portfolio_file.with_suffix(f'.json.backup_{timestamp}')
        try:
            shutil.copy2(portfolio_file, backup_file)
            print(f"   ✅ Бэкап: {backup_file}")
        except Exception as e:
            print(f"   ⚠️ Не удалось создать бэкап: {e}")

        try:
            with open(portfolio_file, 'r', encoding='utf-8') as f:
                old_state = json.load(f)
            old_cash = old_state.get('cash', 0)
            old_positions = len(old_state.get('positions', {}))
            print(f"   Старое: cash={old_cash:,.0f}₽, positions={old_positions}")
        except:
            pass

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
        print(f"   ✅ Сброшен: cash={initial_capital:,.0f}₽, positions=0")
        files_cleaned.append(str(portfolio_file))
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        files_error.append(str(portfolio_file))

    # ===== 2. ОЧИСТКА PRICE_HISTORY.JSON =====
    price_history_file = Path('data/price_history.json')
    print(f"\n📁 История цен: {price_history_file}")

    if price_history_file.exists():
        backup_file = price_history_file.with_suffix(f'.json.backup_{timestamp}')
        try:
            shutil.copy2(price_history_file, backup_file)
            print(f"   ✅ Бэкап: {backup_file}")
        except Exception as e:
            print(f"   ⚠️ Не удалось создать бэкап: {e}")

    try:
        with open(price_history_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        print(f"   ✅ Очищен")
        files_cleaned.append(str(price_history_file))
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        files_error.append(str(price_history_file))

    # ===== 3. УДАЛЕНИЕ ВЕСОВ МОДЕЛИ =====
    weights_file = Path('models/saved_trader/model_weights.pth')
    print(f"\n📁 Веса модели: {weights_file}")

    if weights_file.exists():
        backup_file = weights_file.with_suffix(f'.pth.backup_{timestamp}')
        try:
            shutil.copy2(weights_file, backup_file)
            print(f"   ✅ Бэкап: {backup_file}")
            weights_file.unlink()
            print(f"   ✅ Веса удалены (модель начнёт с нуля)")
            files_cleaned.append(str(weights_file))
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            files_error.append(str(weights_file))
    else:
        print(f"   ⚠️ Файл не найден")

    # ===== 4. УДАЛЕНИЕ ПАМЯТИ МОДЕЛИ =====
    memory_file = Path('models/saved_trader/memory_buffer.pkl')
    print(f"\n📁 Память модели: {memory_file}")

    if memory_file.exists():
        backup_file = memory_file.with_suffix(f'.pkl.backup_{timestamp}')
        try:
            shutil.copy2(memory_file, backup_file)
            print(f"   ✅ Бэкап: {backup_file}")
            memory_file.unlink()
            print(f"   ✅ Память удалена")
            files_cleaned.append(str(memory_file))
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            files_error.append(str(memory_file))
    else:
        print(f"   ⚠️ Файл не найден")

    # ===== 5. УДАЛЕНИЕ MODEL_STATE.JSON (опционально) =====
    state_file = Path('models/saved_trader/model_state.json')
    print(f"\n📁 Состояние модели: {state_file}")

    if state_file.exists():
        backup_file = state_file.with_suffix(f'.json.backup_{timestamp}')
        try:
            shutil.copy2(state_file, backup_file)
            print(f"   ✅ Бэкап: {backup_file}")
            state_file.unlink()
            print(f"   ✅ Состояние удалено (статистика стратегий сброшена)")
            files_cleaned.append(str(state_file))
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            files_error.append(str(state_file))
    else:
        print(f"   ⚠️ Файл не найден")

    # ===== 6. ОЧИСТКА БЭКОФИСА =====
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
            print(f"   ✅ Удалено {count} файлов")
            files_cleaned.append(f"{backoffice_dir} ({count} файлов)")
        else:
            print(f"   ⚠️ Нет файлов")
    else:
        print(f"   ⚠️ Директория не найдена")

    # ===== 7. ОЧИСТКА ДНЕВНЫХ ОТЧЁТОВ =====
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
        print(f"   ✅ Удалено {count} файлов")
        files_cleaned.append(f"daily_report_*.json ({count} файлов)")
    else:
        print(f"   ⚠️ Нет файлов")

    # ===== 8. ОЧИСТКА TRAINING EXPORT =====
    training_export = Path('data/training_export.json')
    if training_export.exists():
        try:
            training_export.unlink()
            print(f"\n📁 Training export: удалён")
            files_cleaned.append(str(training_export))
        except:
            pass

    # ===== 9. ОЧИСТКА LABELED NEWS =====
    labeled_news = Path('data/labeled_news.json')
    if labeled_news.exists():
        try:
            backup_file = labeled_news.with_suffix(f'.json.backup_{timestamp}')
            shutil.copy2(labeled_news, backup_file)
            with open(labeled_news, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print(f"\n📁 Labeled news: очищен")
            files_cleaned.append(str(labeled_news))
        except:
            pass

    # ===== ИТОГ =====
    print("\n" + "=" * 70)
    print("📋 ИТОГ СБРОСА")
    print("=" * 70)

    print(f"\n   Начальный капитал: {initial_capital:,.0f}₽")
    print(f"\n   Очищено/удалено: {len(files_cleaned)}")
    for f in files_cleaned:
        print(f"   ✅ {f}")

    if files_error:
        print(f"\n   Ошибок: {len(files_error)}")
        for f in files_error:
            print(f"   ❌ {f}")

    print(f"\n   Бэкапы сохранены с суффиксом .backup_{timestamp}")

    if not files_error:
        print(f"\n   ✅ СИСТЕМА ПОЛНОСТЬЮ СБРОШЕНА!")
        print(f"   Портфель: {initial_capital:,.0f}₽, 0 позиций")
        print(f"   Модель: веса удалены, память очищена")
        print(f"   Можно запускать: python main.py")
    else:
        print(f"\n   ⚠️ СБРОС ВЫПОЛНЕН С ОШИБКАМИ. Проверьте права доступа.")

    print("\n" + "=" * 70)
    print("✅ СБРОС ЗАВЕРШЁН")
    print("=" * 70)

    return len(files_error) == 0


if __name__ == "__main__":
    print("\n⚠️ ВНИМАНИЕ!")
    print("Этот скрипт ПОЛНОСТЬЮ сбросит систему:")
    print("  — Портфель: 10 000₽, 0 позиций")
    print("  — Историю цен: очищена")
    print("  — Веса модели: УДАЛЕНЫ (обучение с нуля)")
    print("  — Память модели: УДАЛЕНА")
    print("  — Статистика стратегий: СБРОШЕНА")
    print("  — Дневные отчёты: удалены")
    print("  — Бэкапы будут созданы автоматически")

    response = input("\nПродолжить? (yes/no): ").strip().lower()

    if response in ['yes', 'y', 'да', 'д']:
        reset_system()
    else:
        print("❌ Сброс отменён.")