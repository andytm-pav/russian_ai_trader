#!/usr/bin/env python3
"""
Главный модуль запуска AI трейдера для российского рынка
ТЕСТОВАЯ ВЕРСИЯ - игнорирует проверку торгового времени
"""

import signal
import sys
import time
from datetime import datetime
import threading

from core.trading_hours_scheduler import TradingScheduler
from models.smart_broker import SmartPortfolioBroker
from web.app import run_web_server
from utils.logger import setup_logger

logger = setup_logger("MAIN")
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
    """Основная функция запуска - ТЕСТОВАЯ ВЕРСИЯ"""
    logger.info("=" * 60)
    logger.info("ЗАПУСК AI ТРЕЙДЕРА ДЛЯ РОССИЙСКОГО РЫНКА")
    logger.info("ТЕСТОВЫЙ РЕЖИМ - торговля 24/7")
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
    broker = SmartPortfolioBroker(settings)

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

    # Основной цикл работы - ТЕСТОВАЯ ВЕРСИЯ
    logger.info("Система запущена и готова к работе")
    logger.info(f"Начальный капитал: {settings['initial_capital_rub']:,}₽")
    logger.info(f"Макс. позиций: {settings['max_positions']}")
    logger.info(f"Торговые часы: {settings['trading_hours']['main_session']}")
    logger.info("ТЕСТ: проверка времени отключена, торговля 24/7")

    cycle_counter = 0
    while not stop_event.is_set():
        try:
            # +++ ТЕСТ: ВСЕГДА запускаем торговый цикл, игнорируем проверку времени +++
            # if scheduler.is_trading_time():  # ОРИГИНАЛЬНАЯ СТРОКА
            # Запуск торгового цикла
            broker.run_cycle()

            # Периодический лог
            cycle_counter += 1
            if cycle_counter % 5 == 0:  # Логируем чаще для теста
                total_value = broker.portfolio.get_total_value({})
                logger.info(
                    f"ТЕСТ Цикл #{cycle_counter} | Портфель: {total_value:,.0f}₽ | "
                    f"Кэш: {broker.portfolio.cash:,.0f}₽ | "
                    f"Позиций: {len(broker.portfolio.positions)}")

            # +++ ДОПОЛНИТЕЛЬНЫЙ ЛОГ ДЛЯ ПЕРВЫХ ЦИКЛОВ +++
            if cycle_counter <= 3:
                logger.info(f"Первые циклы: #{cycle_counter} выполнен")
                # Принудительно сбрасываем кэш для теста новостей
                try:
                    if hasattr(broker, 'news_core'):
                        broker.news_core.fetch_all_news()
                        logger.info("Принудительный сбор новостей выполнен")
                except Exception as e:
                    logger.error(f"Ошибка принудительного сбора: {e}")

            # Пауза между циклами (всегда 30 секунд в тестовом режиме)
            # pause = 30 if scheduler.is_trading_time() else 60  # ОРИГИНАЛЬНАЯ СТРОКА
            pause = 30  # Всегда 30 секунд в тестовом режиме
            stop_event.wait(pause)

        except KeyboardInterrupt:
            logger.info("Получен Ctrl+C, останавливаю...")
            break
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
            time.sleep(30)  # Меньшая пауза при ошибке в тестовом режиме

    # Graceful shutdown
    logger.info("Завершение работы...")

    # Сохранение состояния
    try:
        broker.model.save_model()
        logger.info("Модель сохранена")
    except Exception as e:
        logger.error(f"Ошибка сохранения модели: {e}")

    logger.info("Работа завершена")


if __name__ == "__main__":
    main()