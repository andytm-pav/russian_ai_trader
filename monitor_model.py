#!/usr/bin/env python3
"""
Мониторинг состояния модели трейдера в реальном времени.
Запускается параллельно с main.py.
"""

import json
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Пути к данным модели
MODEL_DIR = "models/saved_trader"
STATE_PATH = os.path.join(MODEL_DIR, "model_state.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights.pth")
PORTFOLIO_STATE_PATH = "data/portfolio_state.json"
SETTINGS_PATH = "config/settings.json"
STRATEGIES_PATH = "config/strategies.json"


def load_model_state():
    """Загрузить состояние модели из файла"""
    if not os.path.exists(STATE_PATH):
        print("❌ Файл состояния модели не найден!")
        print(f"Ожидаемый путь: {STATE_PATH}")
        return None

    try:
        with open(STATE_PATH, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # Проверяем время последнего сохранения
        if 'save_time' in state:
            save_time = datetime.fromisoformat(state['save_time'].replace('Z', '+00:00'))
            time_diff = datetime.now() - save_time
            state['_metadata'] = {
                'load_time': datetime.now().isoformat(),
                'seconds_since_save': time_diff.total_seconds(),
                'minutes_since_save': time_diff.total_seconds() / 60,
                'hours_since_save': time_diff.total_seconds() / 3600
            }

        return state
    except Exception as e:
        print(f"❌ Ошибка загрузки состояния: {e}")
        return None


def load_portfolio_state():
    """Загрузить состояние портфеля"""
    if not os.path.exists(PORTFOLIO_STATE_PATH):
        return None

    try:
        with open(PORTFOLIO_STATE_PATH, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)

        # Преобразуем positions в список, если это словарь
        if 'positions' in portfolio and isinstance(portfolio['positions'], dict):
            positions_list = []
            for ticker, pos_data in portfolio['positions'].items():
                if isinstance(pos_data, dict):
                    positions_list.append({
                        'ticker': ticker,
                        'quantity': pos_data.get('qty', 0),
                        'avg_price': pos_data.get('avg_price', 0),
                        'buy_time': pos_data.get('buy_time', 0),
                        'position_value': pos_data.get('qty', 0) * pos_data.get('avg_price', 0)
                    })
            portfolio['positions'] = positions_list

        return portfolio
    except Exception as e:
        print(f"⚠ Ошибка загрузки портфеля: {e}")
        return None


def load_settings():
    """Загрузить настройки"""
    if not os.path.exists(SETTINGS_PATH):
        return None

    try:
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Ошибка загрузки настроек: {e}")
        return None


def load_strategies():
    """Загрузить стратегии"""
    if not os.path.exists(STRATEGIES_PATH):
        return None

    try:
        with open(STRATEGIES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Ошибка загрузки стратегий: {e}")
        return None


def analyze_trading_state(state):
    """Анализ состояния торговли"""
    if not state or 'ticker_stats' not in state:
        return {}

    ticker_stats = state['ticker_stats']

    analysis = {
        'summary': {},
        'top_performers': [],
        'worst_performers': [],
        'recent_activity': []
    }

    # Общая статистика
    total_trades = sum(stats.get('total_trades', 0) for stats in ticker_stats.values())
    profitable_trades = sum(stats.get('profitable_trades', 0) for stats in ticker_stats.values())
    total_pnl = sum(stats.get('total_pnl', 0) for stats in ticker_stats.values())

    analysis['summary'] = {
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'total_pnl': total_pnl,
        'unique_tickers': len(ticker_stats),
        'success_rate': profitable_trades / total_trades if total_trades > 0 else 0,
        'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
    }

    # Топ-5 лучших тикеров
    ticker_performance = []
    for ticker, stats in ticker_stats.items():
        if stats.get('total_trades', 0) >= 3:  # Минимум 3 сделки
            success_rate = stats.get('success_rate', 0)
            avg_hold_time = stats.get('avg_hold_time', 0)
            total_pnl = stats.get('total_pnl', 0)

            # Скор с учетом нескольких факторов
            score = (success_rate * 0.4 +
                     min(total_pnl / 1000, 1.0) * 0.3 +  # Нормализованный PnL
                     (1 - min(avg_hold_time / 48, 1.0)) * 0.3)  # Предпочтение быстрым сделкам

            ticker_performance.append({
                'ticker': ticker,
                'score': score,
                'trades': stats.get('total_trades', 0),
                'success_rate': success_rate,
                'total_pnl': total_pnl,
                'avg_hold_time': avg_hold_time
            })

    # Сортировка по скору
    ticker_performance.sort(key=lambda x: x['score'], reverse=True)

    analysis['top_performers'] = ticker_performance[:5]
    analysis['worst_performers'] = ticker_performance[-5:] if len(ticker_performance) >= 5 else []

    # Недавняя активность (последние 24 часа)
    recent_cutoff = datetime.now() - timedelta(hours=24)
    recent_tickers = []

    for ticker, stats in ticker_stats.items():
        last_trade = stats.get('last_trade')
        if last_trade:
            try:
                last_trade_time = datetime.fromisoformat(last_trade.replace('Z', '+00:00'))
                if last_trade_time > recent_cutoff:
                    recent_tickers.append({
                        'ticker': ticker,
                        'last_trade': last_trade,
                        'trades': stats.get('total_trades', 0),
                        'success_rate': stats.get('success_rate', 0)
                    })
            except:
                pass

    # Сортировка по времени последней сделки
    recent_tickers.sort(key=lambda x: x['last_trade'], reverse=True)
    analysis['recent_activity'] = recent_tickers[:10]

    return analysis


def analyze_strategy_performance(state):
    """Анализ эффективности стратегий"""
    if not state or 'strategy_performance' not in state:
        return {}

    strategy_perf = state['strategy_performance']

    analysis = {
        'summary': {},
        'strategies': []
    }

    for strategy, perf in strategy_perf.items():
        total_trades = perf.get('total_trades', 0)

        if total_trades > 0:
            strategy_data = {
                'name': strategy,
                'total_trades': total_trades,
                'profitable_trades': perf.get('profitable_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'total_pnl': perf.get('total_pnl', 0),
                'avg_pnl': perf.get('avg_pnl', 0),
                'efficiency_score': perf.get('win_rate', 0) * perf.get('avg_pnl', 0) * 100
            }
            analysis['strategies'].append(strategy_data)

    # Сортировка по эффективности
    analysis['strategies'].sort(key=lambda x: x['efficiency_score'], reverse=True)

    # Сводка
    if analysis['strategies']:
        total_trades_all = sum(s['total_trades'] for s in analysis['strategies'])
        total_pnl_all = sum(s['total_pnl'] for s in analysis['strategies'])
        avg_win_rate = sum(s['win_rate'] * s['total_trades'] for s in
                           analysis['strategies']) / total_trades_all if total_trades_all > 0 else 0

        analysis['summary'] = {
            'total_strategies': len(analysis['strategies']),
            'total_trades': total_trades_all,
            'total_pnl': total_pnl_all,
            'avg_win_rate': avg_win_rate,
            'best_strategy': analysis['strategies'][0]['name'] if analysis['strategies'] else 'N/A',
            'worst_strategy': analysis['strategies'][-1]['name'] if analysis['strategies'] else 'N/A'
        }

    return analysis


def analyze_error_patterns(state):
    """Анализ паттернов ошибок"""
    if not state or 'error_memory' not in state:
        return {}

    error_memory = state['error_memory']

    analysis = {
        'summary': {},
        'common_patterns': [],
        'high_risk_tickers': []
    }

    # Статистика по ошибкам
    total_failures = 0
    total_avg_loss = 0
    tickers_with_failures = []

    for ticker, error_data in error_memory.items():
        failures = error_data.get('failure_count', 0)
        if failures > 0:
            total_failures += failures
            total_avg_loss += error_data.get('avg_loss', 0)

            tickers_with_failures.append({
                'ticker': ticker,
                'failures': failures,
                'avg_loss': error_data.get('avg_loss', 0),
                'last_failure': error_data.get('last_failure'),
                'success_rate': error_data.get('success_rate', 0),
                'total_trades': error_data.get('total_trades', 0)
            })

    analysis['summary'] = {
        'tickers_with_failures': len(tickers_with_failures),
        'total_failures': total_failures,
        'avg_failures_per_ticker': total_failures / len(tickers_with_failures) if tickers_with_failures else 0,
        'overall_avg_loss': total_avg_loss / len(tickers_with_failures) if tickers_with_failures else 0
    }

    # Сортировка тикеров по количеству ошибок
    tickers_with_failures.sort(key=lambda x: x['failures'], reverse=True)
    analysis['high_risk_tickers'] = tickers_with_failures[:10]

    return analysis


def determine_development_stage(state, settings):
    """Определение стадии развития модели"""
    if not state:
        return "НЕИНИЦИАЛИЗИРОВАНА"

    # Получаем данные
    ticker_stats = state.get('ticker_stats', {})
    total_trades = sum(stats.get('total_trades', 0) for stats in ticker_stats.values())
    memory_size = state.get('memory_size', 0)

    # Критерии для определения стадии
    if total_trades == 0 and memory_size == 0:
        return "🚀 НОВИЧОК: Модель только что создана, сделок нет"

    elif total_trades < 10:
        return f"🎓 УЧЕНИК: {total_trades} сделок, идет накопление опыта"

    elif total_trades < 50:
        return f"📈 НАЧИНАЮЩИЙ: {total_trades} сделок, начальное обучение"

    elif total_trades < 200:
        return f"⚡ АКТИВНЫЙ: {total_trades} сделок, активное обучение"

    elif total_trades < 500:
        return f"🏆 ОПЫТНЫЙ: {total_trades} сделок, стабильная работа"

    elif memory_size >= 3000:
        return f"🤖 ЭКСПЕРТ: {total_trades} сделок, {memory_size} опытов, максимальная память"

    else:
        # Определяем по успешности
        profitable_trades = sum(stats.get('profitable_trades', 0) for stats in ticker_stats.values())
        success_rate = profitable_trades / total_trades if total_trades > 0 else 0

        if success_rate > 0.6:
            return f"✅ УСПЕШНЫЙ: {total_trades} сделок, {success_rate:.1%} успешность"
        elif success_rate > 0.5:
            return f"📊 СТАБИЛЬНЫЙ: {total_trades} сделок, {success_rate:.1%} успешность"
        else:
            return f"🔄 АДАПТИРУЮЩИЙСЯ: {total_trades} сделок, {success_rate:.1%} успешность, требуется оптимизация"


def analyze_market_state(state):
    """Анализ рыночного состояния"""
    if not state:
        return {}

    analysis = {
        'market_metrics': {},
        'trends': {}
    }

    # Базовые метрики
    analysis['market_metrics'] = {
        'market_sentiment': state.get('market_sentiment', 0),
        'volatility_index': state.get('volatility_index', 1.0),
        'sentiment_history_length': len(state.get('sentiment_history', []))
    }

    # Анализ трендов сентимента
    sentiment_history = state.get('sentiment_history', [])
    if len(sentiment_history) >= 10:
        recent_sentiments = sentiment_history[-10:]

        analysis['trends']['sentiment'] = {
            'current': recent_sentiments[-1] if recent_sentiments else 0,
            'avg_last_10': np.mean(recent_sentiments),
            'trend': 'ВОСХОДЯЩИЙ' if len(recent_sentiments) >= 2 and recent_sentiments[-1] > recent_sentiments[
                -2] else 'НИСХОДЯЩИЙ',
            'volatility': np.std(recent_sentiments) if recent_sentiments else 0
        }

    # Интерпретация сентимента
    sentiment = state.get('market_sentiment', 0)
    if sentiment > 0.3:
        analysis['trends']['interpretation'] = "📈 СИЛЬНЫЙ БЫЧИЙ НАСТРОЙ"
    elif sentiment > 0.1:
        analysis['trends']['interpretation'] = "📈 БЫЧИЙ НАСТРОЙ"
    elif sentiment > -0.1:
        analysis['trends']['interpretation'] = "⚖ НЕЙТРАЛЬНЫЙ РЫНОК"
    elif sentiment > -0.3:
        analysis['trends']['interpretation'] = "📉 МЕДВЕЖИЙ НАСТРОЙ"
    else:
        analysis['trends']['interpretation'] = "📉 СИЛЬНЫЙ МЕДВЕЖИЙ НАСТРОЙ"

    return analysis


def calculate_portfolio_pnl(portfolio):
    """Расчет PnL портфеля безопасным способом"""
    if not portfolio or 'positions' not in portfolio:
        return 0

    positions = portfolio.get('positions', [])

    # Если это словарь, конвертируем в список
    if isinstance(positions, dict):
        positions = list(positions.values())

    # Проверяем что каждый элемент - словарь
    total_pnl = 0
    for pos in positions:
        if isinstance(pos, dict):
            total_pnl += pos.get('pnl', 0)
        else:
            # Если это строка или другой тип, пропускаем
            continue

    return total_pnl


def display_dashboard(state, portfolio, settings, strategies):
    """Отображение дашборда"""
    print("\n" + "=" * 80)
    print("🤖 МОНИТОРИНГ МОДЕЛИ AI ТРЕЙДЕРА")
    print("=" * 80)

    # Заголовок с временем
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not state:
        print("\n❌ Модель не найдена или не инициализирована")
        print(f"Проверьте путь: {STATE_PATH}")
        return

    # 1. Стадия развития
    development_stage = determine_development_stage(state, settings)
    print(f"\n📊 СТАДИЯ РАЗВИТИЯ: {development_stage}")

    # 2. Статистика торговли
    trading_analysis = analyze_trading_state(state)
    if trading_analysis.get('summary'):
        summary = trading_analysis['summary']
        print(f"\n📈 СТАТИСТИКА ТОРГОВЛИ:")
        print(f"   Всего сделок: {summary['total_trades']}")
        print(f"   Успешных сделок: {summary['profitable_trades']} ({summary['success_rate']:.1%})")
        print(f"   Общий PnL: {summary['total_pnl']:+.2f}₽")
        print(f"   Средний PnL на сделку: {summary['avg_pnl_per_trade']:+.2f}₽")
        print(f"   Уникальных тикеров: {summary['unique_tickers']}")

    # 3. Рыночное состояние
    market_analysis = analyze_market_state(state)
    if market_analysis.get('market_metrics'):
        metrics = market_analysis['market_metrics']
        print(f"\n🌐 РЫНОЧНОЕ СОСТОЯНИЕ:")
        print(f"   Настроение рынка: {metrics['market_sentiment']:+.3f}")
        print(f"   Индекс волатильности: {metrics['volatility_index']:.2f}")
        if 'trends' in market_analysis and 'interpretation' in market_analysis['trends']:
            print(f"   Интерпретация: {market_analysis['trends']['interpretation']}")

    # 4. Стратегии
    strategy_analysis = analyze_strategy_performance(state)
    if strategy_analysis.get('strategies'):
        print(f"\n🎯 ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ:")
        for i, strategy in enumerate(strategy_analysis['strategies'][:3], 1):
            print(f"   {i}. {strategy['name'].upper()}:")
            print(f"      Сделок: {strategy['total_trades']}, Успешность: {strategy['win_rate']:.1%}")
            print(f"      PnL: {strategy['total_pnl']:+.2f}₽, Средний: {strategy['avg_pnl']:+.2f}₽")

    # 5. Ошибки и риски
    error_analysis = analyze_error_patterns(state)
    if error_analysis.get('summary'):
        summary = error_analysis['summary']
        print(f"\n⚠ ОШИБКИ И РИСКИ:")
        print(f"   Тикеров с ошибками: {summary['tickers_with_failures']}")
        print(f"   Всего ошибок: {summary['total_failures']}")
        if summary['tickers_with_failures'] > 0:
            print(f"   Средний убыток на ошибку: {summary['overall_avg_loss']:.2%}")

    # 6. Портфель
    if portfolio:
        print(f"\n💰 СОСТОЯНИЕ ПОРТФЕЛЯ:")
        print(f"   Общая стоимость: {portfolio.get('total_value', 0):,.0f}₽")
        print(f"   Кэш: {portfolio.get('cash', 0):,.0f}₽")
        print(f"   Позиций: {portfolio.get('positions_count', 0)}")

        # Безопасный расчет PnL
        total_pnl = calculate_portfolio_pnl(portfolio)
        print(f"   Нереализованный PnL: {total_pnl:+,.0f}₽")

    # 7. Системные метрики
    print(f"\n⚙ СИСТЕМНЫЕ МЕТРИКИ:")
    print(f"   Размер памяти: {state.get('memory_size', 0)} опытов")
    print(f"   Опытов ошибок: {state.get('total_experiences', 0)}")
    print(f"   Стратегий в конфиге: {len(strategies.get('strategies', {})) if strategies else 0}")

    if '_metadata' in state:
        metadata = state['_metadata']
        print(f"   Последнее сохранение: {metadata['minutes_since_save']:.1f} мин. назад")

    # 8. Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")

    # На основе анализа
    if trading_analysis.get('summary'):
        success_rate = trading_analysis['summary']['success_rate']

        if success_rate < 0.4:
            print("   ❗ Успешность низкая (<40%). Рассмотрите снижение рисков или изменение стратегий.")
        elif success_rate > 0.6:
            print("   ✅ Успешность высокая (>60%). Можно осторожно увеличивать объемы.")
        else:
            print("   ⚠ Успешность в норме (40-60%). Продолжайте текущую стратегию.")

    if error_analysis.get('high_risk_tickers'):
        high_risk = error_analysis['high_risk_tickers'][:3]
        print("   ⚠ Топ-3 рискованных тикера:")
        for ticker in high_risk:
            print(f"      {ticker['ticker']}: {ticker['failures']} ошибок, успешность {ticker['success_rate']:.1%}")

    # 9. Топ-5 тикеров
    if trading_analysis.get('top_performers'):
        print(f"\n🏆 ТОП-5 ТИКЕРОВ:")
        for i, ticker in enumerate(trading_analysis['top_performers'][:5], 1):
            print(f"   {i}. {ticker['ticker']}: {ticker['trades']} сделок, "
                  f"успешность {ticker['success_rate']:.1%}, PnL {ticker['total_pnl']:+.1f}₽")

    # 10. Недавняя активность
    if trading_analysis.get('recent_activity'):
        recent_count = len(trading_analysis['recent_activity'])
        print(f"\n🔄 НЕДАВНЯЯ АКТИВНОСТЬ (за 24ч): {recent_count} тикеров")
        if recent_count > 0:
            for ticker in trading_analysis['recent_activity'][:3]:
                print(f"   • {ticker['ticker']}: {ticker['trades']} сделок, "
                      f"успешность {ticker['success_rate']:.1%}")

    print("\n" + "=" * 80)
    print("Для обновления нажмите Ctrl+C (остановить) или подождите автообновления...")
    print("=" * 80)


def main():
    """Основная функция мониторинга"""
    print("🚀 Запуск мониторинга модели AI трейдера")
    print("📊 Данные обновляются автоматически каждые 30 секунд")
    print("🛑 Для остановки нажмите Ctrl+C\n")

    update_interval = 30  # секунд
    iteration = 0

    try:
        while True:
            iteration += 1
            print(f"\n🔄 Итерация #{iteration}")
            print(f"⏱ Время: {datetime.now().strftime('%H:%M:%S')}")

            # Загрузка данных
            state = load_model_state()
            portfolio = load_portfolio_state()
            settings = load_settings()
            strategies = load_strategies()

            # Отображение дашборда
            display_dashboard(state, portfolio, settings, strategies)

            # Пауза перед следующим обновлением
            if iteration % 5 == 0:
                print(f"\n⏳ Следующее обновление через {update_interval} секунд...")

            time.sleep(update_interval)

    except KeyboardInterrupt:
        print("\n\n🛑 Мониторинг остановлен пользователем")
        print("📊 Последние данные сохранены в лог")

        # Сохранение последнего состояния в файл
        try:
            last_state = {
                'timestamp': datetime.now().isoformat(),
                'state': load_model_state(),
                'portfolio': load_portfolio_state()
            }

            with open('monitor_last_state.json', 'w', encoding='utf-8') as f:
                json.dump(last_state, f, indent=2, default=str)

            print("💾 Последнее состояние сохранено в monitor_last_state.json")
        except Exception as e:
            print(f"⚠ Не удалось сохранить последнее состояние: {e}")


if __name__ == "__main__":
    main()