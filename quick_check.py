#!/usr/bin/env python3
"""
Простой мониторинг модели трейдера с актуальными ценами
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Импортируем MoexFetcher для получения цен
try:
    from fetchers.moex_fetcher import MoexFetcher

    MOEX_AVAILABLE = True
except ImportError:
    print("⚠ Модуль moex_fetcher не найден, цены будут рассчитаны по цене покупки")
    MOEX_AVAILABLE = False

MODEL_DIR = "models/saved_trader"
STATE_PATH = os.path.join(MODEL_DIR, "model_state.json")
PORTFOLIO_STATE_PATH = "data/portfolio_state.json"


def safe_get(data, key, default=0):
    """Безопасное получение значения"""
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def get_current_prices(tickers: list) -> dict:
    """Получение текущих цен для списка тикеров"""
    prices = {}

    if not MOEX_AVAILABLE or not tickers:
        return prices

    try:
        moex = MoexFetcher()

        # Получаем цены батчами по 50 тикеров
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_prices = moex.get_prices_batch(batch)
            prices.update(batch_prices)

        print(f"   📡 Получены цены для {len(prices)} тикеров с MOEX")

        # Если батч-запрос не сработал, пробуем по одному
        if len(prices) < len(tickers):
            missing = [t for t in tickers if t not in prices]
            print(f"   ⚠ Не получены цены для {len(missing)} тикеров, запрашиваем по одному...")

            for ticker in missing:
                price = moex.get_price(ticker)
                if price:
                    prices[ticker] = price
                    time.sleep(0.1)  # Небольшая задержка

        print(f"   ✅ Итого получены цены для {len(prices)} из {len(tickers)} тикеров")

    except Exception as e:
        print(f"   ❌ Ошибка получения цен: {e}")

    return prices


def calculate_portfolio_value(portfolio_state, current_prices=None):
    """Расчет актуальной стоимости портфеля"""
    if not portfolio_state:
        return 0

    cash = portfolio_state.get('cash', 0)
    positions = portfolio_state.get('positions', {})

    if not isinstance(positions, dict):
        return cash

    total_positions_value = 0

    for ticker, pos in positions.items():
        if not isinstance(pos, dict):
            continue

        qty = pos.get('qty', 0)
        avg_price = pos.get('avg_price', 0)

        # Используем текущую цену если есть, иначе цену покупки
        current_price = None
        if current_prices and ticker in current_prices:
            current_price = current_prices[ticker]
        else:
            current_price = avg_price  # Fallback

        if current_price and current_price > 0:
            total_positions_value += qty * current_price

    return cash + total_positions_value


def analyze_portfolio_details(portfolio_state):
    """Детальный анализ и вывод информации о портфеле с реальными ценами"""
    if not portfolio_state:
        print("   ❌ Данные портфеля не загружены.")
        return

    print(f"\n💰 ПОРТФЕЛЬ (ДЕТАЛИ):")

    # Базовые данные
    cash = portfolio_state.get('cash', 0)
    positions = portfolio_state.get('positions', {})

    if not isinstance(positions, dict):
        print(f"   Общая стоимость: {cash:,.0f}₽")
        print(f"   Наличные (кэш): {cash:,.0f}₽")
        print(f"   Позиций: 0")
        return

    # Получаем список тикеров
    tickers = list(positions.keys())

    print(f"   📡 Получение актуальных цен с MOEX...")
    current_prices = get_current_prices(tickers)

    # Расчет актуальной стоимости
    actual_total_value = calculate_portfolio_value(portfolio_state, current_prices)

    print(f"   Общая стоимость: {actual_total_value:,.0f}₽")
    print(f"   Наличные (кэш): {cash:,.0f}₽")

    # Обработка позиций
    positions_list = []
    for ticker, pos_data in positions.items():
        if isinstance(pos_data, dict):
            positions_list.append({
                'ticker': ticker,
                'quantity': safe_get(pos_data, 'qty', 0),
                'avg_price': safe_get(pos_data, 'avg_price', 0),
                'buy_time': safe_get(pos_data, 'buy_time', 0),
                'strategy': safe_get(pos_data, 'strategy'),
                'stop_loss': safe_get(pos_data, 'stop_loss'),
                'take_profit': safe_get(pos_data, 'take_profit')
            })

    total_positions = len(positions_list)
    print(f"   Всего позиций: {total_positions}")

    if total_positions > 0:
        print(f"\n   📋 СПИСОК ПОЗИЦИЙ (с актуальными ценами):")

        total_cost_basis = 0
        total_market_value = 0
        total_unrealized_pnl = 0

        for i, pos in enumerate(positions_list, 1):
            ticker = pos['ticker']
            qty = pos['quantity']
            avg_price = pos['avg_price']
            strategy = pos['strategy']

            # Получаем текущую цену
            current_price = current_prices.get(ticker, 0)
            if current_price == 0:
                current_price = avg_price  # Если не получили цену, используем цену покупки

            # Расчет метрик
            cost_basis = qty * avg_price
            market_value = qty * current_price
            unrealized_pnl = market_value - cost_basis
            pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0
            weight = (market_value / actual_total_value * 100) if actual_total_value > 0 else 0

            # Аккумулируем общие значения
            total_cost_basis += cost_basis
            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl

            # Форматирование времени покупки
            buy_time_str = "N/A"
            buy_time = pos['buy_time']
            if buy_time:
                try:
                    if isinstance(buy_time, (int, float)):
                        buy_time_str = datetime.fromtimestamp(buy_time).strftime('%d.%m.%Y %H:%M')
                    elif isinstance(buy_time, str):
                        buy_time_str = buy_time[:16].replace('T', ' ')
                except:
                    buy_time_str = str(buy_time)[:16]

            # Определяем источник цены
            price_source = "🔄 MOEX" if ticker in current_prices and current_prices[ticker] > 0 else "📊 AVG"

            # Вывод информации о позиции
            print(f"     {i:2}. {ticker:10} {qty:5} шт.")
            print(f"         Ср.цена: {avg_price:8.2f}₽ | Тек.цена: {current_price:8.2f}₽ ({price_source})")
            print(f"         Стоимость: {market_value:8,.0f}₽ | Вес: {weight:5.1f}%")

            # Цветной вывод PnL
            pnl_sign = "+" if unrealized_pnl >= 0 else ""
            pnl_color = "🟢" if unrealized_pnl > 0 else "🔴" if unrealized_pnl < 0 else "⚪"
            print(f"         PnL: {pnl_color} {pnl_sign}{unrealized_pnl:+,.0f}₽ ({pnl_percent:+.1f}%)")

            # Дополнительная информация
            if strategy:
                print(f"         Стратегия: {strategy}")

            if buy_time_str != "N/A":
                print(f"         Куплено: {buy_time_str}")

            # Стоп-лосс и тейк-профит если есть
            stop_loss = pos.get('stop_loss')
            take_profit = pos.get('take_profit')
            if stop_loss or take_profit:
                sl_info = f"SL: {stop_loss:.1f}₽" if stop_loss else ""
                tp_info = f"TP: {take_profit:.1f}₽" if take_profit else ""
                print(f"         {sl_info} {tp_info}".strip())

        # Сводная статистика по портфелю
        print(f"\n   📊 СВОДКА ПО ПОРТФЕЛЮ:")
        print(f"       Общая стоимость позиций: {total_market_value:,.0f}₽")
        print(f"       Общая стоимость покупки: {total_cost_basis:,.0f}₽")

        total_pnl_sign = "+" if total_unrealized_pnl >= 0 else ""
        total_pnl_color = "🟢" if total_unrealized_pnl > 0 else "🔴" if total_unrealized_pnl < 0 else "⚪"
        total_pnl_percent = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0
        print(
            f"       Суммарный PnL: {total_pnl_color} {total_pnl_sign}{total_unrealized_pnl:+,.0f}₽ ({total_pnl_percent:+.1f}%)")

        # Распределение по стратегиям
        if any(p.get('strategy') for p in positions_list):
            print(f"\n   🎯 РАСПРЕДЕЛЕНИЕ ПО СТРАТЕГИЯМ:")
            strategies = {}
            for pos in positions_list:
                strategy = pos.get('strategy', 'Неизвестно')
                if strategy not in strategies:
                    strategies[strategy] = 0
                strategies[strategy] += pos['quantity'] * current_prices.get(pos['ticker'], pos['avg_price'])

            for strategy, value in strategies.items():
                weight = (value / total_market_value * 100) if total_market_value > 0 else 0
                print(f"       {strategy:20} {weight:5.1f}% ({value:,.0f}₽)")

        # Топ-5 позиций по стоимости (с актуальными ценами)
        if positions_list:
            print(f"\n   🏆 ТОП-ПОЗИЦИЙ ПО СТОИМОСТИ:")
            sorted_positions = sorted(
                positions_list,
                key=lambda x: x['quantity'] * current_prices.get(x['ticker'], x['avg_price']),
                reverse=True
            )

            for i, pos in enumerate(sorted_positions[:5], 1):
                ticker = pos['ticker']
                market_val = pos['quantity'] * current_prices.get(ticker, pos['avg_price'])
                weight = (market_val / actual_total_value * 100) if actual_total_value > 0 else 0
                print(f"       {i}. {ticker:10} {weight:5.1f}% ({market_val:,.0f}₽)")


def analyze_model():
    """Анализ модели"""
    print("\n" + "=" * 70)
    print("🤖 СТАТУС МОДЕЛИ AI ТРЕЙДЕРА")
    print("=" * 70)

    # Проверка файлов
    if not os.path.exists(STATE_PATH):
        print("❌ Модель не найдена!")
        print(f"Путь: {STATE_PATH}")
        return

    try:
        # Загрузка состояния модели
        with open(STATE_PATH, 'r', encoding='utf-8') as f:
            model_state = json.load(f)

        # Загрузка портфеля
        portfolio_state = None
        if os.path.exists(PORTFOLIO_STATE_PATH):
            with open(PORTFOLIO_STATE_PATH, 'r', encoding='utf-8') as f:
                portfolio_state = json.load(f)

        # 1. Основная информация
        print(f"\n📅 ПОСЛЕДНЕЕ СОХРАНЕНИЕ: {model_state.get('save_time', 'Неизвестно')}")

        # 2. Статистика торговли
        ticker_stats = model_state.get('ticker_stats', {})
        total_trades = sum(safe_get(stats, 'total_trades') for stats in ticker_stats.values())
        profitable_trades = sum(safe_get(stats, 'profitable_trades') for stats in ticker_stats.values())
        success_rate = profitable_trades / total_trades if total_trades > 0 else 0

        print(f"\n📊 ТОРГОВАЯ СТАТИСТИКА:")
        print(f"   Всего сделок: {total_trades}")
        print(f"   Успешных: {profitable_trades} ({success_rate:.1%})")
        print(f"   Уникальных тикеров: {len(ticker_stats)}")

        # 3. Память и обучение
        print(f"\n💾 ПАМЯТЬ И ОБУЧЕНИЕ:")
        print(f"   Опытов в памяти: {model_state.get('memory_size', 0)}")
        print(f"   Всего опытов: {model_state.get('total_experiences', 0)}")

        # 4. Стратегии
        strategy_perf = model_state.get('strategy_performance', {})
        if strategy_perf:
            print(f"\n🎯 ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ:")
            for strategy, perf in strategy_perf.items():
                trades = safe_get(perf, 'total_trades')
                if trades > 0:
                    win_rate = safe_get(perf, 'win_rate', 0)
                    avg_pnl = safe_get(perf, 'avg_pnl', 0)
                    total_pnl = safe_get(perf, 'total_pnl', 0)
                    print(f"   • {strategy}: {trades} сделок, {win_rate:.1%} успешность")
                    print(f"     PnL: {total_pnl:+,.0f}₽, Средний: {avg_pnl:+.2f}₽")

        # 5. Рыночное состояние
        print(f"\n🌐 РЫНОЧНОЕ СОСТОЯНИЕ:")
        print(f"   Настроение: {model_state.get('market_sentiment', 0):+.3f}")
        print(f"   Волатильность: {model_state.get('volatility_index', 1.0):.2f}")

        # 6. Ошибки
        error_memory = model_state.get('error_memory', {})
        if error_memory:
            tickers_with_errors = len([v for v in error_memory.values() if safe_get(v, 'failure_count', 0) > 0])
            print(f"\n⚠ ОШИБКИ:")
            print(f"   Тикеров с ошибками: {tickers_with_errors}")

            # Показываем топ-3 тикеров по ошибкам
            error_tickers = []
            for ticker, error_data in error_memory.items():
                failures = safe_get(error_data, 'failure_count', 0)
                if failures > 0:
                    error_tickers.append((ticker, failures, safe_get(error_data, 'avg_loss', 0)))

            error_tickers.sort(key=lambda x: x[1], reverse=True)
            for ticker, failures, avg_loss in error_tickers[:3]:
                print(f"   • {ticker}: {failures} ошибок, средний убыток: {avg_loss:.2%}")

        # 7. ДЕТАЛЬНЫЙ АНАЛИЗ ПОРТФЕЛЯ С АКТУАЛЬНЫМИ ЦЕНАМИ
        analyze_portfolio_details(portfolio_state)

        # 8. Оценка стадии
        print(f"\n📈 СТАДИЯ РАЗВИТИЯ:")
        if total_trades == 0:
            print("   🚀 НОВИЧОК: Модель только что создана")
        elif total_trades < 10:
            print(f"   🎓 УЧЕНИК: {total_trades} сделок, идет накопление опыта")
        elif total_trades < 50:
            print(f"   📈 НАЧИНАЮЩИЙ: {total_trades} сделок, начальное обучение")
        elif total_trades < 200:
            print(f"   ⚡ АКТИВНЫЙ: {total_trades} сделок, активное обучение")
        else:
            print(f"   🏆 ОПЫТНЫЙ: {total_trades} сделок, стабильная работа")

        # 9. Рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if total_trades == 0:
            print("   ✅ Начните торговлю для накопления опыта")
        elif success_rate < 0.4:
            print("   ⚠ Низкая успешность. Рассмотрите снижение рисков")
        elif success_rate > 0.6:
            print("   ✅ Высокая успешность. Можно осторожно увеличивать объемы")
        else:
            print("   📊 Успешность в норме. Продолжайте текущую стратегию")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Ошибка анализа: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Основная функция"""
    print("🔍 Мониторинг модели AI трейдера")
    if MOEX_AVAILABLE:
        print("📡 Используются реальные цены с MOEX API")
    else:
        print("⚠ Внимание: реальные цены недоступны, используются цены покупки")
    print("=" * 50)

    analyze_model()

    while True:
        try:
            print("\n🔄 Для повторного анализа нажмите Enter, для выхода введите 'q'")
            user_input = input(">>> ").strip().lower()

            if user_input == 'q':
                print("\n🛑 Выход из мониторинга")
                break

            print("\n" + "=" * 50)
            analyze_model()

        except KeyboardInterrupt:
            print("\n\n🛑 Мониторинг остановлен")
            break


if __name__ == "__main__":
    main()