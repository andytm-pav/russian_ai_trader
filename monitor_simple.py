#!/usr/bin/env python3
"""
Простой мониторинг модели трейдера с корректной оценкой стадии
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


def analyze_actual_model_progress(model_state, portfolio_state):
    """Анализ реального прогресса модели на основе ВСЕХ данных"""

    analysis = {
        'estimated_total_trades': 0,
        'recent_activity': 0,
        'data_sources': [],
        'confidence': 'низкая'
    }

    # 1. Сделки из ticker_stats (могут быть устаревшими)
    ticker_stats = model_state.get('ticker_stats', {})
    stats_trades = sum(safe_get(stats, 'total_trades') for stats in ticker_stats.values())
    analysis['ticker_stats_trades'] = stats_trades

    # 2. Анализ памяти модели
    memory_size = model_state.get('memory_size', 0)
    analysis['memory_size'] = memory_size

    # Эмпирическая формула: опыт в памяти ≈ 2-3x от сделок
    estimated_trades_from_memory = max(0, memory_size // 2)

    # 3. Анализ стратегий (более актуальные данные)
    strategy_perf = model_state.get('strategy_performance', {})
    strategy_trades = sum(safe_get(perf, 'total_trades', 0) for perf in strategy_perf.values())
    analysis['strategy_trades'] = strategy_trades

    # 4. Анализ ошибок
    error_memory = model_state.get('error_memory', {})
    failed_trades = sum(len(data.get('failed_trades', [])) for data in error_memory.values())
    analysis['failed_trades'] = failed_trades

    # 5. Анализ портфеля (косвенный показатель активности)
    portfolio_activity = 0
    if portfolio_state:
        positions = portfolio_state.get('positions', {})
        if isinstance(positions, dict):
            portfolio_activity = len(positions)
            # Каждая позиция могла быть результатом нескольких сделок
            portfolio_activity *= 2

    # 6. Самый консервативный и реалистичный расчет
    # Берем МАКСИМУМ из всех источников, так как данные могут быть рассинхронизированы
    estimated_total = max(
        stats_trades,  # Из статистики
        strategy_trades,  # Из стратегий
        estimated_trades_from_memory,  # Из памяти
        portfolio_activity,  # Из портфеля
        failed_trades * 3  # Ошибки обычно ~30% сделок
    )

    analysis['estimated_total_trades'] = estimated_total

    # 7. Определяем надежность оценки
    sources_count = sum(1 for x in [stats_trades, strategy_trades, memory_size] if x > 0)
    if sources_count >= 3 and abs(stats_trades - strategy_trades) < 10:
        analysis['confidence'] = 'высокая'
    elif sources_count >= 2:
        analysis['confidence'] = 'средняя'

    # 8. Дополнительные метрики
    analysis['unique_tickers_traded'] = len(ticker_stats)
    analysis['active_strategies'] = len([p for p in strategy_perf.values() if safe_get(p, 'total_trades', 0) > 0])

    # 9. Недавняя активность (по времени сохранения)
    if 'save_time' in model_state:
        try:
            save_time = datetime.fromisoformat(model_state['save_time'].replace('Z', '+00:00'))
            hours_since_save = (datetime.now() - save_time).total_seconds() / 3600
            analysis['hours_since_last_save'] = hours_since_save

            # Если модель сохранялась недавно, значит она активна
            if hours_since_save < 2:
                analysis['recent_activity'] = 1
        except:
            pass

    return analysis


def determine_real_development_stage(analysis):
    """Определение РЕАЛЬНОЙ стадии развития на основе комплексного анализа"""

    estimated_trades = analysis['estimated_total_trades']
    memory_size = analysis['memory_size']
    confidence = analysis['confidence']

    # Основные критерии стадии
    if estimated_trades == 0 and memory_size == 0:
        return "🚀 НОВИЧОК: Модель только что создана"

    elif estimated_trades < 5:
        return f"🎓 САМЫЙ НАЧИНАЮЩИЙ: ~{estimated_trades} сделок (уверенность: {confidence})"

    elif estimated_trades < 20:
        return f"📈 НАЧИНАЮЩИЙ: ~{estimated_trades} сделок, {memory_size} опытов в памяти"

    elif estimated_trades < 100:
        return f"⚡ АКТИВНЫЙ УЧЕНИК: ~{estimated_trades} сделок, идет активное обучение"

    elif estimated_trades < 300:
        return f"🏆 ОПЫТНЫЙ ТРЕЙДЕР: ~{estimated_trades} сделок, стабильная работа"

    elif memory_size >= 3000:
        return f"🤖 ЭКСПЕРТ: ~{estimated_trades} сделок, {memory_size} опытов (макс. память)"

    elif estimated_trades >= 500:
        return f"🚀 ПРОДВИНУТЫЙ ЭКСПЕРТ: ~{estimated_trades} сделок, обширный опыт"

    else:
        # Динамическая оценка по эффективности
        if analysis.get('active_strategies', 0) >= 3:
            return f"🎯 СТРАТЕГИЧЕСКИЙ ТРЕЙДЕР: {analysis['active_strategies']} активных стратегий"
        elif analysis.get('unique_tickers_traded', 0) >= 10:
            return f"📊 РАЗНООБРАЗНЫЙ: {analysis['unique_tickers_traded']} уникальных тикеров"
        else:
            return f"📈 РАЗВИВАЮЩИЙСЯ: ~{estimated_trades} сделок (уверенность: {confidence})"


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

        # 2. Комплексный анализ реального прогресса
        progress_analysis = analyze_actual_model_progress(model_state, portfolio_state)

        print(f"\n📊 РЕАЛЬНЫЙ ПРОГРЕСС МОДЕЛИ:")
        print(f"   Статистика сделок (ticker_stats): {progress_analysis.get('ticker_stats_trades', 0)}")
        print(f"   Сделок по стратегиям: {progress_analysis.get('strategy_trades', 0)}")
        print(f"   Опытов в памяти: {progress_analysis.get('memory_size', 0)}")
        print(f"   Неудачных сделок: {progress_analysis.get('failed_trades', 0)}")
        print(f"   Уникальных тикеров: {progress_analysis.get('unique_tickers_traded', 0)}")
        print(f"   Активных стратегий: {progress_analysis.get('active_strategies', 0)}")

        # 3. Реальная стадия развития
        print(f"\n📈 СТАДИЯ РАЗВИТИЯ:")
        real_stage = determine_real_development_stage(progress_analysis)
        print(f"   {real_stage}")

        # 4. Дополнительные метрики надежности
        estimated_total = progress_analysis['estimated_total_trades']
        print(f"   📈 ОЦЕНОЧНО ВСЕГО СДЕЛОК: ~{estimated_total}")
        print(f"   🎯 УВЕРЕННОСТЬ ОЦЕНКИ: {progress_analysis['confidence']}")

        if 'hours_since_last_save' in progress_analysis:
            hours = progress_analysis['hours_since_last_save']
            if hours < 1:
                print(f"   ⚡ АКТИВНОСТЬ: модель сохранялась {hours:.1f} ч. назад - В РАБОТЕ!")
            elif hours < 24:
                print(f"   ⏱ АКТИВНОСТЬ: модель сохранялась {hours:.1f} ч. назад")
            else:
                print(f"   💤 АКТИВНОСТЬ: модель не сохранялась {hours:.1f} ч.")

        # 5. Рыночное состояние
        print(f"\n🌐 РЫНОЧНОЕ СОСТОЯНИЕ:")
        print(f"   Настроение: {model_state.get('market_sentiment', 0):+.3f}")
        print(f"   Волатильность: {model_state.get('volatility_index', 1.0):.2f}")

        # 6. Стратегии (только если есть данные)
        strategy_perf = model_state.get('strategy_performance', {})
        if strategy_perf:
            active_strategies = [(k, v) for k, v in strategy_perf.items() if safe_get(v, 'total_trades', 0) > 0]
            if active_strategies:
                print(f"\n🎯 ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ:")
                for strategy, perf in active_strategies[:3]:  # Показываем топ-3
                    trades = safe_get(perf, 'total_trades')
                    win_rate = safe_get(perf, 'win_rate', 0)
                    avg_pnl = safe_get(perf, 'avg_pnl', 0)
                    total_pnl = safe_get(perf, 'total_pnl', 0)
                    print(f"   • {strategy}: {trades} сделок, {win_rate:.1%} успешность")
                    if total_pnl != 0:
                        print(f"     PnL: {total_pnl:+,.0f}₽, Средний: {avg_pnl:+.2f}₽")

        # 7. ДЕТАЛЬНЫЙ АНАЛИЗ ПОРТФЕЛЯ С АКТУАЛЬНЫМИ ЦЕНАМИ
        analyze_portfolio_details(portfolio_state)

        # 8. Рекомендации на основе реального прогресса
        print(f"\n💡 РЕКОМЕНДАЦИИ:")

        if progress_analysis['estimated_total_trades'] < 20:
            print("   ✅ Продолжайте накопление опыта, модель еще учится")
        elif progress_analysis['confidence'] == 'низкая':
            print("   ⚠ Данные о сделках могут быть неполными. Проверьте логи работы модели.")
        elif progress_analysis['memory_size'] >= 3000:
            print("   🧠 Модель достигла максимального объема памяти. Рассмотрите очистку старых опытов.")
        elif estimated_total >= 100:
            print("   🎯 Модель имеет значительный опыт. Можно тестировать более сложные стратегии.")
        else:
            print("   📊 Модель в стадии активного обучения. Следите за эффективностью стратегий.")

        # 9. Диагностика возможных проблем
        print(f"\n🔍 ДИАГНОСТИКА:")

        # Проверка синхронизации данных
        stats_trades = progress_analysis.get('ticker_stats_trades', 0)
        strategy_trades = progress_analysis.get('strategy_trades', 0)

        if stats_trades > 0 and strategy_trades > 0 and abs(stats_trades - strategy_trades) > 10:
            print("   ⚠ ВНИМАНИЕ: данные о сделках в разных источниках расходятся!")
            print(f"     • ticker_stats: {stats_trades} сделок")
            print(f"     • strategy_performance: {strategy_trades} сделок")
            print("     Это может указывать на проблему синхронизации данных модели.")

        # Проверка активности модели
        if 'hours_since_last_save' in progress_analysis and progress_analysis['hours_since_last_save'] > 24:
            print("   ⚠ Модель не сохранялась более 24 часов. Возможно, она не работает.")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Ошибка анализа: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Основная функция"""
    print("🔍 Мониторинг модели AI трейдера")
    print("📡 Оценка РЕАЛЬНОГО прогресса на основе всех данных")
    print("=" * 70)

    analyze_model()

    while True:
        try:
            print("\n🔄 Для повторного анализа нажмите Enter, для выхода введите 'q'")
            user_input = input(">>> ").strip().lower()

            if user_input == 'q':
                print("\n🛑 Выход из мониторинга")
                break

            print("\n" + "=" * 70)
            analyze_model()

        except KeyboardInterrupt:
            print("\n\n🛑 Мониторинг остановлен")
            break


if __name__ == "__main__":
    main()