#!/usr/bin/env python3
"""
Главный модуль запуска AI трейдера для российского рынка
"""

import signal
import sys
import time
from datetime import datetime
import threading

from core.trading_hours_scheduler import TradingScheduler
from models.smart_broker import SmartPortfolioBroker
from web.app import run_web_server
from utils.logger import get_logger

logger = get_logger("MAIN")
stop_event = threading.Event()


def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    logger.info(f"Получен сигнал {signum}, останавливаю систему...")
    stop_event.set()


def load_configuration():
    """Загрузка конфигурации"""
    import json
    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        with open('config/broker.json', 'r', encoding='utf-8') as f:
            broker_config = json.load(f)
        return settings, broker_config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        sys.exit(1)


def main():
    """Основная функция запуска"""
    logger.info("=" * 60)
    logger.info("ЗАПУСК AI ТРЕЙДЕРА ДЛЯ РОССИЙСКОГО РЫНКА")
    logger.info("=" * 60)

    # Загрузка конфигурации
    settings, broker_config = load_configuration()

    # Настройка обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Инициализация компонентов
    logger.info("Инициализация компонентов...")

    # 1. Инициализация планировщика торговых сессий
    scheduler = TradingScheduler()

    # 2. Инициализация SmartBroker
    broker = SmartPortfolioBroker(settings, scheduler)

    # 3. Запуск веб-сервера в отдельном потоке
    web_thread = threading.Thread(
        target=run_web_server,
        args=(broker, stop_event),
        daemon=True
    )
    web_thread.start()
    logger.info("Веб-интерфейс запущен на http://localhost:8050")

    # 4. Предсессионный анализ
    broker.pre_session_analysis()

    # Основной цикл работы
    logger.info("Система запущена и готова к работе")
    logger.info(f"Начальный капитал: {settings['initial_capital_rub']:,}₽")
    logger.info(f"Макс. позиций: {settings['max_positions']}")
    logger.info(f"Торговые часы: {settings['trading_hours']['main_session']}")

    cycle_counter = 0
    last_cycle_time = 0

    while not stop_event.is_set():
        try:
            # Получаем текущее московское время для диагностики
            now_moscow = datetime.now(scheduler.moscow_tz)
            is_trading = scheduler.is_trading_time()

            # Диагностический лог перед проверкой торгового времени
            logger.debug(
                f"🔍 [MAIN] {now_moscow.strftime('%H:%M:%S')} | is_trading={is_trading} | cycle={cycle_counter} | stop_event={stop_event.is_set()}")

            if is_trading:
                cycle_num = cycle_counter + 1
                cycle_start = time.time()

                logger.info(f"🟢 [MAIN] Запуск цикла #{cycle_num} в {now_moscow.strftime('%H:%M:%S')}")

                try:
                    # Запуск торгового цикла
                    broker.run_cycle()

                    cycle_time = time.time() - cycle_start
                    logger.info(f"🔴 [MAIN] Цикл #{cycle_num} завершен за {cycle_time:.1f}с")

                    # Увеличиваем счётчик только после успешного выполнения
                    cycle_counter += 1
                    last_cycle_time = time.time()

                    # Периодический лог (каждые 10 циклов)
                    if cycle_counter % 10 == 0:
                        total_value = broker.portfolio.get_total_value({})
                        logger.info(
                            f"📊 СТАТИСТИКА | Цикл #{cycle_counter} | Портфель: {total_value:,.0f}₽ | Кэш: {broker.portfolio.cash:,.0f}₽ | Позиций: {len(broker.portfolio.positions)}")

                except Exception as cycle_error:
                    cycle_time = time.time() - cycle_start
                    logger.error(f"❌ [MAIN] ОШИБКА в цикле #{cycle_num} после {cycle_time:.1f}с: {cycle_error}")
                    import traceback
                    logger.error(f"❌ [MAIN] Детали ошибки:\n{traceback.format_exc()}")
                    # Не увеличиваем cycle_counter при ошибке
                    # Пауза перед следующей попыткой
                    time.sleep(60)
                    continue
            else:
                # Рынок закрыт - диагностический лог раз в минуту
                if int(time.time()) % 60 == 0:
                    logger.debug(f"⏸ [MAIN] Рынок закрыт, пропускаю цикл. Время: {now_moscow.strftime('%H:%M:%S')}")

            # Пауза между циклами
            pause = 10 if scheduler.is_trading_time() else 60

            # Диагностика перед сном
            if cycle_counter > 0 and time.time() - last_cycle_time > 60:
                logger.debug(f"💤 [MAIN] Долгая пауза ({time.time() - last_cycle_time:.0f}с), засыпаю на {pause}с")
            else:
                logger.debug(f"💤 [MAIN] Засыпаю на {pause} секунд")

            # Ожидание с возможностью прерывания
            stop_event.wait(pause)

            # Диагностика после пробуждения
            if not stop_event.is_set():
                logger.debug(f"⏰ [MAIN] Проснулся после сна")

        except KeyboardInterrupt:
            logger.info("🔴 Получен Ctrl+C, останавливаю систему...")
            break
        except Exception as e:
            logger.error(f"❌ [MAIN] Критическая ошибка в основном цикле: {e}")
            import traceback
            logger.error(f"❌ [MAIN] Детали ошибки:\n{traceback.format_exc()}")
            logger.info("⏱ Ожидание 60 секунд перед следующей попыткой...")
            time.sleep(60)

    # Graceful shutdown
    logger.info("=" * 60)
    logger.info("🛑 НАЧАЛО ЗАВЕРШЕНИЯ РАБОТЫ СИСТЕМЫ")
    logger.info("=" * 60)

    # Останавливаем торговлю
    broker.trading_enabled = False
    logger.info("⏸ Торговля остановлена")

    # Завершаем работу брокера (сохраняет состояние, закрывает позиции)
    try:
        broker.shutdown()
        logger.info("✅ Брокер успешно завершил работу")
    except Exception as e:
        logger.error(f"❌ Ошибка при завершении брокера: {e}")

    # Сохранение модели
    try:
        broker.model.save_model()
        logger.info("✅ Модель сохранена")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения модели: {e}")

    # Сохранение истории портфеля
    try:
        broker._save_portfolio_state()
        logger.info("✅ Состояние портфеля сохранено")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения портфеля: {e}")

    # Останавливаем фоновые потоки
    if hasattr(broker, 'trainer') and broker.trainer:
        try:
            broker.trainer.stop_background_training()
            logger.info("✅ Фоновое обучение остановлено")
        except Exception as e:
            logger.error(f"❌ Ошибка остановки обучения: {e}")

    # Останавливаем планировщик задач
    if hasattr(broker, 'scheduler') and broker.scheduler:
        try:
            broker.scheduler.stop_scheduler()
            logger.info("✅ Планировщик остановлен")
        except Exception as e:
            logger.error(f"❌ Ошибка остановки планировщика: {e}")

    logger.info("=" * 60)
    logger.info("✅ СИСТЕМА УСПЕШНО ЗАВЕРШИЛА РАБОТУ")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()