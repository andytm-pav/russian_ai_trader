#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Очистка мусорных позиций (qty не кратно lot_size) — интерактивная версия.

Запуск:
  python scripts/cleanup_fractional_positions.py

Скрипт сам спросит: dry-run или apply.
"""
import json
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

# ============================================================
# НАСТРОЙКИ ПУТЕЙ (авто-поиск)
# ============================================================
# Скрипт пытается найти portfolio_state.json и tickers.json в нескольких местах
SEARCH_PATHS = [
    Path.cwd(),                              # текущая директория
    Path.cwd().parent,                       # родительская
    Path(__file__).parent.parent if '__file__' in dir() else Path.cwd(),  # ../ от scripts/
    Path.home() / 'PycharmProjects' / 'russian_ai_trader',
]

PORTFOLIO_FILE = None
TICKERS_FILE = None

for base in SEARCH_PATHS:
    p1 = base / 'data' / 'portfolio_state.json'
    p2 = base / 'config' / 'tickers.json'
    if p1.exists() and p2.exists():
        PORTFOLIO_FILE = p1
        TICKERS_FILE = p2
        break

# Если не нашли — спросим у пользователя
if PORTFOLIO_FILE is None:
    print("Не найдены файлы проекта автоматически.")
    print("Укажите путь к папке проекта (где лежат data/ и config/):")
    user_path = input(">>> ").strip().strip('"').strip("'")
    if user_path:
        base = Path(user_path)
        p1 = base / 'data' / 'portfolio_state.json'
        p2 = base / 'config' / 'tickers.json'
        if p1.exists() and p2.exists():
            PORTFOLIO_FILE = p1
            TICKERS_FILE = p2
        else:
            print(f"❌ Файлы не найдены в {user_path}")
            print(f"   Ищу: {p1}")
            print(f"   Ищу: {p2}")
            sys.exit(1)
    else:
        print("❌ Путь не указан.")
        sys.exit(1)

# Лотности по умолчанию (если нет в tickers.json)
DEFAULT_LOTS = {
    'SBER': 1, 'SBERP': 1, 'GAZP': 10, 'LKOH': 1, 'ROSN': 1,
    'NVTK': 1, 'TATN': 10, 'TATNP': 10, 'MTLR': 10, 'MTLRP': 10,
    'GMKN': 10, 'MGNT': 1, 'AFLT': 10, 'VTBR': 10000000,
    'RUAL': 10, 'NLMK': 10, 'CHMF': 10, 'PHOR': 10, 'MOEX': 10,
    'SNGS': 100, 'SNGSP': 100, 'BSPB': 1000, 'YDEX': 1,
    'WUSH': 1, 'VKCO': 1, 'ENPG': 1, 'FESH': 10, 'MTLRP': 10,
    'RNFT': 1, 'RAGR': 1, 'T': 1, 'AFLT': 10, 'TATNP': 10,
}

# Комиссия Т-Банка
COMMISSION_RATE = 0.003
MIN_COMMISSION = 0.01


def load_ticker_lots() -> dict:
    """Загружает справочник лотностей из tickers.json с fallback на DEFAULT_LOTS."""
    lots = dict(DEFAULT_LOTS)
    try:
        with open(TICKERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data.get('watchlist', []):
            t = item['ticker']
            l = item.get('lot_size', 1)
            if t and l:
                lots[t] = l
    except Exception as e:
        print(f"⚠️  Не удалось загрузить tickers.json: {e}")
        print(f"   Использую значения по умолчанию.")
    return lots


def analyze_positions(portfolio: dict, ticker_lots: dict) -> list:
    """Анализирует все позиции на кратность лотам."""
    positions = portfolio.get('positions', {})
    results = []

    for ticker, pos in positions.items():
        try:
            qty = float(pos.get('qty', 0))
        except (ValueError, TypeError):
            qty = 0.0
        lot = ticker_lots.get(ticker, 1)
        try:
            avg_price = float(pos.get('avg_price', 0))
        except (ValueError, TypeError):
            avg_price = 0.0

        if lot <= 1:
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'OK', 'action': 'none', 'reason': 'lot=1 (любое qty)'
            })
            continue

        # Проверяем кратность (с tolerance для float)
        if abs(qty % lot) < 1e-9 or abs(qty % lot - lot) < 1e-9:
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'OK', 'action': 'none', 'reason': f'кратно лоту ({int(qty/lot)} лотов)'
            })
            continue

        # Проблема: qty не кратно lot
        if qty < lot:
            # Меньше 1 лота — мусорная позиция
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'GARBAGE', 'action': 'sell_all',
                'sell_qty': qty,
                'reason': f'qty {qty} < lot {lot} — ПРОДАТЬ ВСЁ'
            })
        else:
            # Больше 1 лота, но не кратно
            full_lots = int(qty // lot)
            remainder = qty - full_lots * lot
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'FRACTIONAL', 'action': 'sell_to_multiple',
                'sell_qty': remainder,
                'keep_qty': full_lots * lot,
                'reason': f'{full_lots}×lot + {remainder:.4f} — продать остаток'
            })

    return results


def apply_cleanup(portfolio: dict, analysis: list, dry_run: bool = True) -> dict:
    """Применяет очистку к портфелю."""
    positions = portfolio.get('positions', {})
    cash = float(portfolio.get('cash', 0))
    trade_history = portfolio.get('trade_history', [])

    changes = []
    total_pnl = 0.0
    total_commission = 0.0
    total_revenue = 0.0

    for item in analysis:
        if item['action'] == 'none':
            continue

        ticker = item['ticker']
        qty_to_sell = item['sell_qty']

        if ticker not in positions:
            continue

        pos = positions[ticker]
        avg_price = item['avg_price']
        # Текущая цена — из last_price позиции, либо из avg, либо спросим пользователя
        current_price = pos.get('last_price')
        if current_price is None or current_price <= 0:
            # Пробуем взять из позиции (некоторые хранят current_price)
            current_price = pos.get('current_price', avg_price)

        revenue = qty_to_sell * current_price
        commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
        pnl = (current_price - avg_price) * qty_to_sell - commission

        changes.append({
            'ticker': ticker,
            'action': 'SELL',
            'qty_sold': qty_to_sell,
            'price': current_price,
            'revenue': revenue,
            'commission': commission,
            'pnl': pnl,
            'remaining_qty': pos.get('qty', 0) - qty_to_sell,
        })

        total_pnl += pnl
        total_commission += commission
        total_revenue += revenue

        if not dry_run:
            # Обновляем позицию
            remaining = pos.get('qty', 0) - qty_to_sell
            if remaining < 1e-9:
                # Полностью закрываем
                if ticker in positions:
                    del positions[ticker]
            else:
                # Частичная продажа
                pos['qty'] = remaining
                # Уменьшаем commission_buy пропорционально
                old_comm = pos.get('commission_buy', 0)
                if old_comm > 0 and pos.get('qty', 0) > 0:
                    pos['commission_buy'] = old_comm * (remaining / (remaining + qty_to_sell))

            # Обновляем кэш
            cash += revenue - commission

            # Записываем в историю
            trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'ticker': ticker,
                'quantity': qty_to_sell,
                'price': current_price,
                'revenue': revenue,
                'commission': commission,
                'commission_sell': commission,
                'pnl': pnl,
                'pnl_percent': (pnl / (qty_to_sell * avg_price) * 100) if avg_price > 0 else 0,
                'strategy': pos.get('strategy', 'cleanup'),
                'reason': f'cleanup_fractional lot={item["lot"]}',
                'cash_after': cash,
            })

    portfolio['positions'] = positions
    portfolio['cash'] = cash
    portfolio['trade_history'] = trade_history

    return {
        'changes': changes,
        'total_pnl': total_pnl,
        'total_commission': total_commission,
        'total_revenue': total_revenue,
        'portfolio': portfolio,
    }


def print_analysis(analysis: list):
    """Выводит таблицу анализа позиций."""
    print()
    print("АНАЛИЗ ПОЗИЦИЙ:")
    print(f"{'Тикер':7} | {'qty':>10} | {'lot':>6} | {'статус':11} | {'действие':40}")
    print("-" * 85)
    for item in analysis:
        action_desc = item['reason']
        if item['action'] == 'sell_to_multiple':
            action_desc = f"продать {item['sell_qty']:.4f} (останется {item['keep_qty']:.4f})"
        elif item['action'] == 'sell_all':
            action_desc = f"ПРОДАТЬ ВСЁ ({item['qty']:.4f} шт)"
        qty_str = f"{item['qty']:.4f}" if item['qty'] != int(item['qty']) else f"{int(item['qty'])}"
        lot_str = f"{item['lot']}"
        print(f"{item['ticker']:7} | {qty_str:>10} | {lot_str:>6} | "
              f"{item['status']:11} | {action_desc}")


def print_changes(result: dict):
    """Выводит список изменений."""
    print()
    print("ИЗМЕНЕНИЯ:")
    if not result['changes']:
        print("  ✅ Нет позиций для очистки — все позиции корректны!")
        return

    for c in result['changes']:
        print(f"  {c['ticker']:7} SELL {c['qty_sold']:>10.4f} @ {c['price']:.2f} "
              f"= {c['revenue']:.2f}₽ (comm {c['commission']:.2f}₽, P&L {c['pnl']:+.2f}₽, "
              f"останется {c['remaining_qty']:.4f})")

    print()
    print(f"  Всего сделок: {len(result['changes'])}")
    print(f"  Сумма выручки: {result['total_revenue']:.2f}₽")
    print(f"  Сумма комиссий: {result['total_commission']:.2f}₽")
    print(f"  Чистый P&L: {result['total_pnl']:+.2f}₽")


def main():
    print("=" * 85)
    print("ОЧИСТКА МУСОРНЫХ ПОЗИЦИЙ (qty не кратно lot_size)")
    print("=" * 85)
    print()
    print(f"Найден файл портфеля: {PORTFOLIO_FILE}")
    print(f"Найден файл тикеров:  {TICKERS_FILE}")
    print()

    # Загружаем данные
    try:
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка загрузки portfolio_state.json: {e}")
        sys.exit(1)

    ticker_lots = load_ticker_lots()

    print(f"Кэш: {float(portfolio.get('cash', 0)):.2f}₽")
    print(f"Позиций: {len(portfolio.get('positions', {}))}")
    print(f"Загружено лотностей: {len(ticker_lots)} тикеров")

    # Анализ
    analysis = analyze_positions(portfolio, ticker_lots)
    print_analysis(analysis)

    # Проверяем, есть ли что очищать
    has_changes = any(item['action'] != 'none' for item in analysis)
    if not has_changes:
        print()
        print("✅ Все позиции корректны — очистка не требуется.")
        return

    # Считаем предполагаемые изменения (dry-run)
    result = apply_cleanup(portfolio, analysis, dry_run=True)
    print_changes(result)

    # Интерактивный выбор
    print()
    print("=" * 85)
    print("ВЫБОР РЕЖИМА:")
    print("=" * 85)
    print()
    print("  1) Dry-run — только показать (без изменений)")
    print("  2) APPLY   — применить очистку (с созданием backup)")
    print("  3) Выход   — отмена")
    print()
    choice = input("Введите номер [1/2/3]: ").strip()

    if choice == '1':
        print()
        print("ℹ️  Dry-run режим — изменения НЕ применены.")
        print("   Для применения запустите снова и выберите 2.")
        return
    elif choice == '3':
        print("Отмена.")
        return
    elif choice != '2':
        print(f"Неизвестный выбор: {choice}")
        return

    # Подтверждение перед APPLY
    print()
    print("⚠️  ВНИМАНИЕ! Будут внесены изменения в portfolio_state.json:")
    print(f"   - Продано позиций: {len(result['changes'])}")
    print(f"   - Сумма выручки: {result['total_revenue']:.2f}₽")
    print(f"   - Сумма комиссий: {result['total_commission']:.2f}₽")
    print(f"   - Чистый P&L: {result['total_pnl']:+.2f}₽")
    print()
    confirm = input("Подтвердите выполнение [y/N]: ").strip().lower()
    if confirm not in ('y', 'yes', 'д', 'да'):
        print("Отмена пользователем.")
        return

    # Создаём backup
    backup_dir = PORTFOLIO_FILE.parent / 'backups' / 'cleanup'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / f"portfolio_before_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        shutil.copy2(PORTFOLIO_FILE, backup_file)
        print(f"✅ Backup создан: {backup_file}")
    except Exception as e:
        print(f"⚠️  Не удалось создать backup: {e}")
        print("   Продолжаю без backup...")
        backup_file = None

    # Применяем очистку
    result_apply = apply_cleanup(portfolio, analysis, dry_run=False)

    # Сохраняем
    try:
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(result_apply['portfolio'], f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Портфель обновлён: {PORTFOLIO_FILE}")
        print()
        print("ИТОГ:")
        print(f"  Кэш после очистки: {float(result_apply['portfolio']['cash']):.2f}₽")
        print(f"  Позиций осталось: {len(result_apply['portfolio']['positions'])}")
        print(f"  Чистый P&L: {result_apply['total_pnl']:+.2f}₽")
        if backup_file:
            print()
            print(f"  Backup: {backup_file}")
            print(f"  Для отката скопируйте backup обратно в {PORTFOLIO_FILE}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        if backup_file:
            print(f"   Восстановите из backup: {backup_file}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Прервано пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
