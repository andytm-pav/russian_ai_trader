#!/usr/bin/env python3
"""
Главный модуль запуска AI трейдера для российского рынка
"""

import signal
import sys
import time
# from datetime import datetime
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

    # Основной цикл работы
    logger.info("Система запущена и готова к работе")
    logger.info(f"Начальный капитал: {settings['initial_capital_rub']:,}₽")
    logger.info(f"Макс. позиций: {settings['max_positions']}")
    logger.info(f"Торговые часы: {settings['trading_hours']['main_session']}")

    cycle_counter = 0
    while not stop_event.is_set():
        try:
            # Проверка торгового времени
            if scheduler.is_trading_time():
                # Запуск торгового цикла
                broker.run_cycle()

                # Периодический лог
                cycle_counter += 1
                if cycle_counter % 10 == 0:
                    total_value = broker.portfolio.get_total_value({})
                    logger.info(
                        f"Цикл #{cycle_counter} | Портфель: {total_value:,.0f}₽ | Кэш: {broker.portfolio.cash:,.0f}₽")

            # Пауза между циклами (30 секунд в торговое время, 60 секунд в не торговое)
            pause = 30 if scheduler.is_trading_time() else 60
            stop_event.wait(pause)

        except KeyboardInterrupt:
            logger.info("Получен Ctrl+C, останавливаю...")
            break
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
            time.sleep(60)

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