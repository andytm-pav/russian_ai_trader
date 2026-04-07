#!/usr/bin/env python3
"""
Диагностика работающей системы без остановки.
Подключается к тому же коду и читает состояние планировщика.
"""

import sys
import os
import time
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trading_hours_scheduler import TradingScheduler
from utils.logger import get_logger

logger = get_logger("ATTACH_DEBUG")


def debug_scheduler():
    """Диагностика планировщика в реальном времени"""

    # Создаём НОВЫЙ экземпляр (читает те же конфиги)
    scheduler = TradingScheduler()

    print("\n" + "=" * 80)
    print("🔍 ДИАГНОСТИКА ПЛАНИРОВЩИКА (работает параллельно)")
    print("=" * 80)

    while True:
        try:
            now_moscow = datetime.now(scheduler.moscow_tz)

            # 1. Базовая проверка
            is_trading_day = scheduler.is_trading_day()
            is_trading_time = scheduler.is_trading_time()
            period = scheduler.get_current_moex_period()
            can_trade = scheduler.can_trade_now()

            # 2. Детальная проверка времени
            current_time = now_moscow.time()
            current_hour = current_time.hour
            current_minute = current_time.minute

            # 3. Загружаем конфиг
            import json
            with open("config/market_schedule.json", "r") as f:
                config = json.load(f)

            sessions = config.get('sessions', {})
            main_session = sessions.get('main_session', {})
            evening_session = sessions.get('evening_session', {})

            # 4. Парсим время сессий
            main_start = main_session.get('start', 'N/A')
            main_end = main_session.get('end', 'N/A')
            evening_start = evening_session.get('start', 'N/A')
            evening_end = evening_session.get('end', 'N/A')
            evening_enabled = evening_session.get('enabled', False)

            # 5. Проверяем попадание в интервалы
            in_main = False
            in_evening = False

            if main_start != 'N/A':
                sh, sm = map(int, main_start.split(':'))
                eh, em = map(int, main_end.split(':'))
                from datetime import time as dt_time
                main_start_t = dt_time(sh, sm)
                main_end_t = dt_time(eh, em)
                in_main = main_start_t <= current_time <= main_end_t

            if evening_enabled and evening_start != 'N/A':
                sh, sm = map(int, evening_start.split(':'))
                eh, em = map(int, evening_end.split(':'))
                evening_start_t = dt_time(sh, sm)
                evening_end_t = dt_time(eh, em)
                in_evening = evening_start_t <= current_time <= evening_end_t

            # 6. ВЫВОД
            print(f"\n{'=' * 60}")
            print(f"🕐 Время МСК: {now_moscow.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 60}")
            print(f"📅 Торговый день: {is_trading_day}")
            print(f"💰 Торговое время (is_trading_time): {is_trading_time}")
            print(f"📊 Текущий период MOEX: {period}")
            print(f"✅ Можно торговать: {can_trade.get('can_place_orders', False)}")
            print(f"🎯 Текущий период из can_trade: {can_trade.get('current_period', 'unknown')}")

            print(f"\n📋 СЕССИИ ИЗ КОНФИГА:")
            print(f"   Основная: {main_start} - {main_end} → текущее время ВНУТРИ: {in_main}")
            print(f"   Вечерняя: enabled={evening_enabled}, {evening_start} - {evening_end} → внутри: {in_evening}")

            # 7. Предупреждение, если не совпадает
            if is_trading_time != (in_main or in_evening):
                print(
                    f"\n⚠️ ВНИМАНИЕ: is_trading_time()={is_trading_time}, но по интервалам должно быть {in_main or in_evening}!")
                print(f"   → Проблема в логике is_trading_time() или is_trading_day()")

            if not is_trading_time and (in_main or in_evening):
                print(f"\n🔴 КРИТИЧЕСКАЯ ОШИБКА: Рынок должен быть открыт, но is_trading_time()=False!")
                print(f"   → Проверьте метод is_trading_day() и праздники")

            # Пауза между проверками
            time.sleep(5)

        except KeyboardInterrupt:
            print("\nДиагностика остановлена")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)


def check_main_process():
    """Проверка, жив ли основной процесс main.py"""
    import psutil
    import subprocess

    print("\n" + "=" * 60)
    print("🔍 ПРОВЕРКА ПРОЦЕССОВ")
    print("=" * 60)

    # Ищем процесс main.py
    main_pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'main.py' in cmdline and 'python' in cmdline.lower():
                main_pids.append(proc.info['pid'])
                print(f"✅ Найден процесс main.py: PID={proc.info['pid']}")
                print(f"   Команда: {cmdline[:200]}")

                # Статус потока
                try:
                    proc_cpu = proc.cpu_percent(interval=0.5)
                    proc_mem = proc.memory_info().rss / 1024 / 1024
                    print(f"   CPU: {proc_cpu}%, RAM: {proc_mem:.1f}MB")
                    print(f"   Статус: {proc.status()}")
                except:
                    pass
        except:
            pass

    if not main_pids:
        print("❌ Процесс main.py НЕ НАЙДЕН! Система не запущена.")

    print("=" * 60)


if __name__ == "__main__":
    # Сначала проверяем процессы
    check_main_process()

    # Затем запускаем диагностику планировщика
    print("\n" + "=" * 80)
    print("ЗАПУСК ДИАГНОСТИКИ ПЛАНИРОВЩИКА (нажмите Ctrl+C для остановки)")
    print("=" * 80)

    try:
        debug_scheduler()
    except KeyboardInterrupt:
        print("\nДиагностика завершена")