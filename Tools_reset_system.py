#!/usr/bin/env python3
"""
СБРОС И ВОССТАНОВЛЕНИЕ СИСТЕМЫ (v4)
Позволяет сбросить или восстановить: память, веса, портфель, модель, всё сразу.
"""

import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')


def backup_file(file_path: Path, timestamp: str) -> bool:
    """Создать бэкап файла"""
    if not file_path.exists():
        return False
    backup = file_path.with_suffix(f'{file_path.suffix}.backup_{timestamp}')
    try:
        shutil.copy2(file_path, backup)
        print(f"   ✅ Бэкап: {backup}")
        return True
    except Exception as e:
        print(f"   ⚠️ Не удалось создать бэкап: {e}")
        return False


def list_backups(file_path: Path) -> list:
    """Показать список бэкапов для файла"""
    pattern = f"{file_path.name}.backup_*"
    backups = sorted(file_path.parent.glob(pattern), reverse=True)
    return backups


def restore_file(file_path: Path, backup_path: Path) -> bool:
    """Восстановить файл из бэкапа"""
    if not backup_path.exists():
        print(f"   ❌ Бэкап не найден: {backup_path}")
        return False
    try:
        shutil.copy2(backup_path, file_path)
        print(f"   ✅ Восстановлено: {file_path}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка восстановления: {e}")
        return False


def choose_backup(file_path: Path, description: str) -> Path:
    """Выбрать бэкап из списка"""
    backups = list_backups(file_path)
    if not backups:
        print(f"   ⚠️ Нет бэкапов для {description}")
        return None

    print(f"\n📁 Бэкапы {description}:")
    for i, b in enumerate(backups[:10], 1):
        size = b.stat().st_size
        print(f"   {i}. {b.name} ({size:,} байт)")
    print(f"   0. Отмена")

    while True:
        choice = input("   Выберите номер (0-10): ").strip()
        if choice == '0':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(backups[:10]):
                return backups[idx]
        except:
            pass
        print("   ❌ Неверный выбор.")


def reset_portfolio(initial_capital: float, timestamp: str):
    """Сброс портфеля"""
    portfolio_file = Path('data/portfolio_state.json')
    print(f"\n📁 Портфель: {portfolio_file}")

    if portfolio_file.exists():
        backup_file(portfolio_file, timestamp)

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
        'daily_profit_history': [],
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
        return True
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def restore_portfolio():
    """Восстановить портфель из бэкапа"""
    portfolio_file = Path('data/portfolio_state.json')
    backup = choose_backup(portfolio_file, "портфеля")
    if backup:
        restore_file(portfolio_file, backup)


def reset_price_history(timestamp: str):
    """Очистка истории цен"""
    price_file = Path('data/price_history.json')
    print(f"\n📁 История цен: {price_file}")
    if price_file.exists():
        backup_file(price_file, timestamp)
    try:
        with open(price_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        print(f"   ✅ Очищена")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def restore_price_history():
    """Восстановить историю цен из бэкапа"""
    price_file = Path('data/price_history.json')
    backup = choose_backup(price_file, "истории цен")
    if backup:
        restore_file(price_file, backup)


def reset_memory(timestamp: str):
    """Удаление памяти модели"""
    memory_file = Path('models/saved_trader/memory_buffer.pkl')
    print(f"\n📁 Память модели: {memory_file}")
    if memory_file.exists():
        backup_file(memory_file, timestamp)
        try:
            memory_file.unlink()
            print(f"   ✅ Удалена")
            return True
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return False
    else:
        print(f"   ⚠️ Файл не найден")
        return True


def restore_memory():
    """Восстановить память модели из бэкапа"""
    memory_file = Path('models/saved_trader/memory_buffer.pkl')
    backup = choose_backup(memory_file, "памяти модели")
    if backup:
        restore_file(memory_file, backup)


def reset_weights(timestamp: str):
    """Удаление весов модели"""
    weights_file = Path('models/saved_trader/model_weights.pth')
    print(f"\n📁 Веса модели: {weights_file}")
    if weights_file.exists():
        backup_file(weights_file, timestamp)
        try:
            weights_file.unlink()
            print(f"   ✅ Удалены (обучение с нуля)")
            return True
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return False
    else:
        print(f"   ⚠️ Файл не найден")
        return True


def restore_weights():
    """Восстановить веса модели из бэкапа"""
    weights_file = Path('models/saved_trader/model_weights.pth')
    backup = choose_backup(weights_file, "весов модели")
    if backup:
        restore_file(weights_file, backup)


def reset_model_state(timestamp: str):
    """Удаление состояния модели"""
    state_file = Path('models/saved_trader/model_state.json')
    print(f"\n📁 Состояние модели: {state_file}")
    if state_file.exists():
        backup_file(state_file, timestamp)
        try:
            state_file.unlink()
            print(f"   ✅ Удалено (статистика стратегий сброшена)")
            return True
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return False
    else:
        print(f"   ⚠️ Файл не найден")
        return True


def restore_model_state():
    """Восстановить состояние модели из бэкапа"""
    state_file = Path('models/saved_trader/model_state.json')
    backup = choose_backup(state_file, "состояния модели")
    if backup:
        restore_file(state_file, backup)


def reset_daily_reports(timestamp: str):
    """Очистка дневных отчётов"""
    data_dir = Path('data')
    print(f"\n📁 Дневные отчёты: {data_dir}")
    count = 0
    for pattern in ['daily_report_*.json', 'portfolio_history.json', 'training_export.json']:
        for f in data_dir.glob(pattern):
            try:
                f.unlink()
                count += 1
            except:
                pass
    backoffice = Path('data/backoffice')
    if backoffice.exists():
        for f in backoffice.glob('*.json'):
            try:
                f.unlink()
                count += 1
            except:
                pass
    print(f"   ✅ Удалено {count} файлов")
    return True


def reset_labeled_news(timestamp: str):
    """Очистка размеченных новостей"""
    news_file = Path('data/labeled_news.json')
    print(f"\n📁 Labeled news: {news_file}")
    if news_file.exists():
        backup_file(news_file, timestamp)
        try:
            with open(news_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print(f"   ✅ Очищены")
            return True
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return False
    else:
        print(f"   ⚠️ Файл не найден")
        return True


def restore_labeled_news():
    """Восстановить размеченные новости из бэкапа"""
    news_file = Path('data/labeled_news.json')
    backup = choose_backup(news_file, "новостей")
    if backup:
        restore_file(news_file, backup)


def show_menu():
    """Показать меню выбора"""
    print("\n" + "=" * 60)
    print("🔄 СБРОС И ВОССТАНОВЛЕНИЕ СИСТЕМЫ (v4)")
    print("=" * 60)
    print("  СБРОС:")
    print("    1. Портфель (cash=10000, positions=0)")
    print("    2. Историю цен")
    print("    3. Память модели (memory_buffer.pkl)")
    print("    4. Веса модели (model_weights.pth)")
    print("    5. Состояние модели (model_state.json)")
    print("    6. Дневные отчёты и историю")
    print("    7. Размеченные новости")
    print("    8. ВСЁ (полный сброс)")
    print("  ВОССТАНОВИТЬ:")
    print("    11. Портфель из бэкапа")
    print("    12. Историю цен из бэкапа")
    print("    13. Память модели из бэкапа")
    print("    14. Веса модели из бэкапа")
    print("    15. Состояние модели из бэкапа")
    print("    17. Размеченные новости из бэкапа")
    print("    18. ВСЁ из последних бэкапов")
    print("    0. Выход")
    print("-" * 60)


def main():
    """Точка входа"""
    initial_capital = 10000.0
    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
            initial_capital = settings.get('initial_capital_rub', 10000.0)
    except:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        show_menu()
        choice = input("Ваш выбор: ").strip()

        if choice == '0':
            print("👋 Выход.")
            break

        # Сброс
        elif choice == '1':
            reset_portfolio(initial_capital, timestamp)
        elif choice == '2':
            reset_price_history(timestamp)
        elif choice == '3':
            reset_memory(timestamp)
        elif choice == '4':
            reset_weights(timestamp)
        elif choice == '5':
            reset_model_state(timestamp)
        elif choice == '6':
            reset_daily_reports(timestamp)
        elif choice == '7':
            reset_labeled_news(timestamp)
        elif choice == '8':
            print("\n⚠️ ПОЛНЫЙ СБРОС ВСЕХ ДАННЫХ!")
            confirm = input("Подтвердите (yes/no): ").strip().lower()
            if confirm in ['yes', 'y', 'да', 'д']:
                reset_portfolio(initial_capital, timestamp)
                reset_price_history(timestamp)
                reset_memory(timestamp)
                reset_weights(timestamp)
                reset_model_state(timestamp)
                reset_daily_reports(timestamp)
                reset_labeled_news(timestamp)
                print(f"\n✅ ПОЛНЫЙ СБРОС ЗАВЕРШЁН!")

        # Восстановление
        elif choice == '11':
            restore_portfolio()
        elif choice == '12':
            restore_price_history()
        elif choice == '13':
            restore_memory()
        elif choice == '14':
            restore_weights()
        elif choice == '15':
            restore_model_state()
        elif choice == '17':
            restore_labeled_news()
        elif choice == '18':
            print("\n⚠️ ПОЛНОЕ ВОССТАНОВЛЕНИЕ ИЗ ПОСЛЕДНИХ БЭКАПОВ!")
            confirm = input("Подтвердите (yes/no): ").strip().lower()
            if confirm in ['yes', 'y', 'да', 'д']:
                restore_portfolio()
                restore_price_history()
                restore_memory()
                restore_weights()
                restore_model_state()
                restore_labeled_news()
                print(f"\n✅ ПОЛНОЕ ВОССТАНОВЛЕНИЕ ЗАВЕРШЕНО!")
        else:
            print("❌ Неверный выбор.")

    print("\n" + "=" * 60)
    print("✅ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()