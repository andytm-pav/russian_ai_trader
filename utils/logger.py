"""
Настраиваемая система логирования для торговой системы
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
import os

from collections import deque

# Кольцевой буфер логов для дашборда (последние 500 строк)
_log_buffer = deque(maxlen=500)

# Уровни логирования
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Цвета для консольного вывода (опционально)
COLORS = {
    'DEBUG': '\033[94m',  # Синий
    'INFO': '\033[92m',  # Зеленый
    'WARNING': '\033[93m',  # Желтый
    'ERROR': '\033[91m',  # Красный
    'CRITICAL': '\033[95m',  # Пурпурный
    'RESET': '\033[0m'  # Сброс
}


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом"""

    def format(self, record):
        # Форматируем сообщение стандартным способом
        formatted_message = super().format(record)

        # Добавляем цвет к ВСЕЙ строке, а не к частям
        if record.levelname in COLORS:
            return f"{COLORS[record.levelname]}{formatted_message}{COLORS['RESET']}"

        return formatted_message


class TradeLogger:
    """Кастомный логгер для торговой системы"""

    def __init__(self,
                 name: str,
                 log_level: str = 'INFO',
                 log_to_file: bool = True,
                 log_file: str = 'logs/trading.log',
                 max_file_size: int = 10 * 1024 * 1024,  # 10 MB
                 backup_count: int = 5):

        self.name = name
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_file = log_file

        # Создаем директорию для логов если нужно
        if log_to_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Настройка логгера
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVELS.get(log_level, logging.INFO))
        self.logger.propagate = False

        # Очистка существующих обработчиков
        self.logger.handlers.clear()

        # Настройка формата
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        colored_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS.get(log_level, logging.INFO))
        console_handler.setFormatter(colored_formatter)
        self.logger.addHandler(console_handler)

        # Файловый обработчик
        if log_to_file:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(LOG_LEVELS.get(log_level, logging.INFO))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # JSON логгер для структурированных данных
        self.json_logger = None
        if log_to_file:
            json_log_file = log_file.replace('.log', '_json.log')
            self.json_logger = JsonFileLogger(json_log_file, max_file_size, backup_count)

    def log(self, level: str, message: str, *args, **kwargs):
        """Основной метод логирования"""
        extra_data = kwargs.pop('extra', {})

        # Добавляем дополнительную информацию
        if extra_data:
            message = f"{message} | {json.dumps(extra_data, ensure_ascii=False)}"

        # Добавляем в кольцевой буфер для дашборда
        _log_buffer.append({
            'timestamp': datetime.now().isoformat(),
            'name': self.name,
            'level': level,
            'message': message
        })


        # Логируем в стандартный логгер
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message, *args, **kwargs)

        # Логируем в JSON если нужно
        if self.json_logger and level in ['INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.json_logger.log(level, message, extra_data)

    def debug(self, message: str, *args, **kwargs):
        """Логирование уровня DEBUG"""
        self.log('DEBUG', message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Логирование уровня INFO"""
        self.log('INFO', message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Логирование уровня WARNING"""
        self.log('WARNING', message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Логирование уровня ERROR"""
        self.log('ERROR', message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Логирование уровня CRITICAL"""
        self.log('CRITICAL', message, *args, **kwargs)

    def trade(self,
              action: str,
              ticker: str,
              quantity: int,
              price: float,
              pnl: float = 0.0,
              **kwargs):
        """Специальный метод для логирования сделок"""
        trade_data = {
            'action': action,
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        message = f"TRADE: {action} {quantity}×{ticker} @ {price:.2f}"
        if pnl != 0:
            message += f" | PnL: {pnl:+,.0f}₽"

        self.info(message, extra=trade_data)

    def signal(self,
               ticker: str,
               signal_type: str,
               confidence: float,
               reason: str,
               **kwargs):
        """Специальный метод для логирования сигналов"""
        signal_data = {
            'ticker': ticker,
            'signal_type': signal_type,
            'confidence': confidence,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        message = f"SIGNAL: {signal_type} {ticker} (conf: {confidence:.2f}) - {reason}"
        self.info(message, extra=signal_data)

    def performance(self,
                    metric: str,
                    value: float,
                    period: str = 'daily',
                    **kwargs):
        """Специальный метод для логирования производительности"""
        perf_data = {
            'metric': metric,
            'value': value,
            'period': period,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        message = f"PERF: {metric} = {value:.2f} ({period})"
        self.info(message, extra=perf_data)


class JsonFileLogger:
    """Логгер для структурированных JSON данных"""

    def __init__(self,
                 log_file: str,
                 max_file_size: int,
                 backup_count: int):

        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        from logging.handlers import RotatingFileHandler

        self.handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )

    def log(self, level: str, message: str, extra_data: Dict[str, Any]):
        """Логирование в JSON формате"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'data': extra_data
        }

        log_line = json.dumps(log_entry, ensure_ascii=False) + '\n'

        try:
            self.handler.stream.write(log_line)
            self.handler.stream.flush()
        except Exception as e:
            print(f"Ошибка записи JSON лога: {e}")


def get_log_buffer() -> deque:
    """Получение кольцевого буфера логов для дашборда"""
    return _log_buffer


def clear_log_buffer():
    """Очистка буфера логов"""
    _log_buffer.clear()

# Глобальные логгеры для разных компонентов
_loggers = {}


def setup_logger(name: str,
                 log_level: str = 'INFO',
                 log_to_file: bool = True) -> TradeLogger:
    """Настройка и получение логгера"""
    if name not in _loggers:
        log_file = f"logs/{name.lower()}.log"
        _loggers[name] = TradeLogger(
            name=name,
            log_level=log_level,
            log_to_file=log_to_file,
            log_file=log_file
        )

    return _loggers[name]


def get_logger(name: str, settings_path: str = "config/settings.json") -> TradeLogger:
    """Умный логгер, автоматически загружающий уровень из settings.json"""
    log_level = "INFO"  # Значение по умолчанию

    try:
        # Пробуем загрузить настройки из settings.json
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                log_level = settings.get("logging_level", "INFO")
        else:
            print(f"⚠ Файл настроек не найден: {settings_path}")
    except Exception as e:
        print(f"⚠ Ошибка загрузки настроек логирования: {e}")

    print(f"📝 Логгер '{name}' инициализирован с уровнем: {log_level}")
    return setup_logger(name, log_level=log_level)


# Глобальный логгер с автоматической загрузкой настроек
logger = get_logger('SYSTEM')  # ← ЭТО ЕДИНСТВЕННАЯ СТРОКА создания глобального логгера!


def get_all_loggers() -> Dict[str, TradeLogger]:
    """Получение всех логгеров"""
    return _loggers.copy()


def set_global_log_level(level: str):
    """Установка глобального уровня логирования"""
    for logger_instance in _loggers.values():
        logger_instance.logger.setLevel(LOG_LEVELS.get(level, logging.INFO))
        for handler in logger_instance.logger.handlers:
            handler.setLevel(LOG_LEVELS.get(level, logging.INFO))

    logger.info(f"Глобальный уровень логирования установлен: {level}")