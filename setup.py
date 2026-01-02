"""
Скрипт установки и настройки AI Trader
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Версия установщика
INSTALLER_VERSION = "1.0.0"
MIN_PYTHON_VERSION = (3, 8)


class AITraderInstaller:
    """Установщик и настройщик AI Trader"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models" / "saved_trader"

        # Цвета для консольного вывода
        self.COLORS = {
            'HEADER': '\033[95m',
            'OKBLUE': '\033[94m',
            'OKCYAN': '\033[96m',
            'OKGREEN': '\033[92m',
            'WARNING': '\033[93m',
            'FAIL': '\033[91m',
            'ENDC': '\033[0m',
            'BOLD': '\033[1m',
        }

    def print_header(self):
        """Печать заголовка установщика"""
        print(f"{self.COLORS['HEADER']}{'=' * 60}")
        print("🤖 AI TRADER - УСТАНОВЩИК И НАСТРОЙЩИК")
        print(f"{'=' * 60}{self.COLORS['ENDC']}")
        print(f"Версия установщика: {INSTALLER_VERSION}")
        print()

    def check_python_version(self) -> bool:
        """Проверка версии Python"""
        print(f"{self.COLORS['OKBLUE']}[1/8] Проверка версии Python...{self.COLORS['ENDC']}")

        if sys.version_info < MIN_PYTHON_VERSION:
            print(f"{self.COLORS['FAIL']}❌ Требуется Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+")
            print(f"Текущая версия: {sys.version_info.major}.{sys.version_info.minor}{self.COLORS['ENDC']}")
            return False

        print(
            f"{self.COLORS['OKGREEN']}✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}{self.COLORS['ENDC']}")
        return True

    def create_directories(self):
        """Создание необходимых директорий"""
        print(f"{self.COLORS['OKBLUE']}[2/8] Создание структуры директорий...{self.COLORS['ENDC']}")

        directories = [
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.project_root / "static",
            self.project_root / "web" / "templates"
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"  📁 Создана: {directory.relative_to(self.project_root)}")
            except Exception as e:
                print(f"{self.COLORS['WARNING']}  ⚠ Не удалось создать {directory}: {e}{self.COLORS['ENDC']}")

        print(f"{self.COLORS['OKGREEN']}✅ Структура директорий создана{self.COLORS['ENDC']}")

    def install_dependencies(self):
        """Установка зависимостей Python"""
        print(f"{self.COLORS['OKBLUE']}[3/8] Установка зависимостей Python...{self.COLORS['ENDC']}")

        requirements_file = self.project_root / "requirements.txt"

        if not requirements_file.exists():
            print(f"{self.COLORS['FAIL']}❌ Файл requirements.txt не найден{self.COLORS['ENDC']}")
            return False

        try:
            print("  📦 Установка пакетов (это может занять несколько минут)...")

            # Используем pip для установки
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"{self.COLORS['FAIL']}❌ Ошибка установки зависимостей:{self.COLORS['ENDC']}")
                print(result.stderr)
                return False

            print(f"{self.COLORS['OKGREEN']}✅ Зависимости установлены{self.COLORS['ENDC']}")
            return True

        except Exception as e:
            print(f"{self.COLORS['FAIL']}❌ Ошибка: {e}{self.COLORS['ENDC']}")
            return False

    def create_default_configs(self):
        """Создание конфигурационных файлов по умолчанию"""
        print(f"{self.COLORS['OKBLUE']}[4/8] Создание конфигурационных файлов...{self.COLORS['ENDC']}")

        configs = {
            "settings.json": {
                "initial_capital_rub": 10000,
                "max_positions": 5,
                "max_position_weight_percent": 20,
                "risk_per_trade_percent": 1.5,
                "daily_loss_limit_percent": 5,
                "trading_hours": {
                    "main_session": "06:50-18:50",
                    "evening_session": "19:00-23:50",
                    "weekend_trading_enabled": false
                },
                "news_sensitivity_score": 0.7,
                "enable_technical_core": true,
                "enable_news_core": true,
                "min_cash_per_trade": 1000,
                "target_positions": 4,
                "stop_loss_percent": 3.0,
                "take_profit_percent": 6.0,
                "commission_percent": 0.05,
                "slippage_percent": 0.1,
                "model_confidence_threshold": 0.65,
                "rss_update_interval_seconds": 60,
                "price_update_interval_seconds": 30,
                "portfolio_rebalance_hours": [10, 14, 16],
                "logging_level": "INFO",
                "simulation_mode": true,
                "paper_trading": true,
                "web_port": 8050
            },
            "broker.json": {
                "broker_name": "paper",
                "paper_trading": true,
                "commission_percent": 0.05,
                "slippage_percent": 0.1,
                "order_settings": {
                    "default_order_type": "limit",
                    "time_in_force": "day",
                    "use_wlim": true,
                    "price_offset_percent": 0.1,
                    "max_spread_percent": 0.5
                }
            },
            "rss_sources.json": {
                "sources": [
                    {
                        "name": "Мосбиржа Новости",
                        "url": "https://www.moex.com/ns/news.rss",
                        "category": "market",
                        "priority": 10,
                        "enabled": true
                    },
                    {
                        "name": "РБК Финансы",
                        "url": "https://www.rbc.ru/finances.rss",
                        "category": "finance",
                        "priority": 9,
                        "enabled": true
                    },
                    {
                        "name": "Интерфакс Финансы",
                        "url": "https://www.interfax.ru/rss/finance.asp",
                        "category": "finance",
                        "priority": 8,
                        "enabled": true
                    }
                ],
                "update_interval_minutes": 5,
                "max_news_per_source": 20,
                "keywords_filter": {
                    "include": ["акци", "дивиденд", "отчет", "прибыль", "выручк", "результат", "прогноз", "рекомендац",
                                "купить", "продать"],
                    "exclude": ["крипто", "криптовалют", "биткоин"]
                }
            },
            "tickers.json": {
                "watchlist": [
                    {
                        "ticker": "SBER",
                        "name": "Сбербанк",
                        "sector": "финансы",
                        "lot_size": 10,
                        "min_step": 0.01,
                        "avg_daily_volume": 50000000,
                        "priority": 10
                    },
                    {
                        "ticker": "GAZP",
                        "name": "Газпром",
                        "sector": "нефтегаз",
                        "lot_size": 10,
                        "min_step": 0.01,
                        "avg_daily_volume": 40000000,
                        "priority": 9
                    },
                    {
                        "ticker": "LKOH",
                        "name": "Лукойл",
                        "sector": "нефтегаз",
                        "lot_size": 1,
                        "min_step": 1,
                        "avg_daily_volume": 1000000,
                        "priority": 9
                    },
                    {
                        "ticker": "YNDX",
                        "name": "Яндекс",
                        "sector": "IT",
                        "lot_size": 1,
                        "min_step": 0.02,
                        "avg_daily_volume": 2000000,
                        "priority": 8
                    },
                    {
                        "ticker": "VTBR",
                        "name": "ВТБ",
                        "sector": "финансы",
                        "lot_size": 10000,
                        "min_step": 0.00005,
                        "avg_daily_volume": 1000000000,
                        "priority": 8
                    }
                ],
                "sector_limits": {
                    "финансы": 40,
                    "нефтегаз": 30,
                    "металлы": 20,
                    "IT": 15,
                    "other": 10
                }
            },
            "market_schedule.json": {
                "trading_calendar": {
                    "regular_days": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "short_days": [],
                    "pre_holiday_early_close": ["2026-02-20", "2026-05-08", "2026-11-03"]
                },
                "sessions": {
                    "main_session": {
                        "start": "06:50",
                        "end": "18:50"
                    },
                    "evening_session": {
                        "start": "19:00",
                        "end": "23:50",
                        "enabled": false
                    }
                },
                "holidays_2026": [
                    "2026-01-01", "2026-01-02", "2026-01-07", "2026-02-23",
                    "2026-03-08", "2026-05-01", "2026-05-09", "2026-06-12",
                    "2026-11-04"
                ]
            },
            "portfolio_config.json": {
                "portfolio_name": "AI_Trader_Russian_Market",
                "strategy_type": "hybrid_news_technical",
                "risk_profile": "moderate",
                "rebalancing": {
                    "schedule": "weekly",
                    "day_of_week": "monday",
                    "time": "10:00",
                    "threshold_percent": 5,
                    "max_turnover_percent": 20
                },
                "cash_management": {
                    "min_cash_percent": 10,
                    "max_cash_percent": 30,
                    "emergency_reserve_rub": 1000
                }
            }
        }

        created_count = 0
        for filename, config in configs.items():
            config_path = self.config_dir / filename

            if not config_path.exists():
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    print(f"  📄 Создан: config/{filename}")
                    created_count += 1
                except Exception as e:
                    print(f"{self.COLORS['WARNING']}  ⚠ Не удалось создать {filename}: {e}{self.COLORS['ENDC']}")
            else:
                print(f"  📄 Существует: config/{filename} (не перезаписывается)")

        print(f"{self.COLORS['OKGREEN']}✅ Создано {created_count} конфигурационных файлов{self.COLORS['ENDC']}")

    def create_initial_data(self):
        """Создание начальных данных"""
        print(f"{self.COLORS['OKBLUE']}[5/8] Создание начальных данных...{self.COLORS['ENDC']}")

        # Портфель
        portfolio_data = {
            "total_value": 10000.0,
            "cash": 10000.0,
            "positions": {},
            "initial_capital": 10000.0,
            "trade_history": [],
            "last_update": "2024-01-15T10:30:00",
            "stats": {
                "cash": 10000.0,
                "positions_count": 0,
                "total_trades": 0,
                "initial_capital": 10000.0,
                "current_capital": 10000.0,
                "last_update": "2024-01-15T10:30:00"
            }
        }

        portfolio_path = self.data_dir / "portfolio_state.json"
        if not portfolio_path.exists():
            with open(portfolio_path, 'w', encoding='utf-8') as f:
                json.dump(portfolio_data, f, indent=2, ensure_ascii=False)
            print(f"  💼 Создан начальный портфель: {portfolio_path.name}")

        # Размеченные новости
        news_data = {
            "news_items": [
                {
                    "title": "Сбербанк объявил о рекордной прибыли за квартал",
                    "summary": "Сбербанк сообщил о росте чистой прибыли на 35%",
                    "source": "РБК Финансы",
                    "category": "finance",
                    "timestamp": "2024-01-15T09:30:00",
                    "tickers": ["SBER"],
                    "sentiment": 0.8,
                    "labels": {
                        "action": "BUY",
                        "confidence": 0.85,
                        "impact_level": "high"
                    }
                }
            ],
            "metadata": {
                "total_news": 1,
                "last_update": "2024-01-15T12:00:00",
                "sources_count": 1,
                "tickers_covered": ["SBER"],
                "average_sentiment": 0.8,
                "labeling_method": "manual",
                "version": "1.0"
            }
        }

        news_path = self.data_dir / "labeled_news.json"
        if not news_path.exists():
            with open(news_path, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, indent=2, ensure_ascii=False)
            print(f"  📰 Созданы размеченные новости: {news_path.name}")

        print(f"{self.COLORS['OKGREEN']}✅ Начальные данные созданы{self.COLORS['ENDC']}")

    def setup_model_weights(self):
        """Настройка весов модели"""
        print(f"{self.COLORS['OKBLUE']}[6/8] Настройка модели AI...{self.COLORS['ENDC']}")

        # Создаем начальные веса модели если их нет
        model_weights_path = self.models_dir / "model_weights.pth"
        model_state_path = self.models_dir / "model_state.json"

        if not model_weights_path.exists():
            print("  🤖 Модель будет обучена при первом запуске")

            # Создаем пустое состояние модели
            initial_state = {
                "error_memory": {},
                "ticker_stats": {},
                "market_sentiment": 0.0,
                "sentiment_history": [],
                "volatility_index": 1.0,
                "memory_size": 0,
                "total_experiences": 0,
                "save_time": "2024-01-15T10:30:00"
            }

            try:
                with open(model_state_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_state, f, indent=2, ensure_ascii=False)
                print(f"  📝 Создано начальное состояние модели")
            except Exception as e:
                print(f"{self.COLORS['WARNING']}  ⚠ Не удалось создать состояние модели: {e}{self.COLORS['ENDC']}")
        else:
            print(f"  🤖 Найдены сохраненные веса модели")

        print(f"{self.COLORS['OKGREEN']}✅ Модель AI настроена{self.COLORS['ENDC']}")

    def create_startup_scripts(self):
        """Создание скриптов для запуска"""
        print(f"{self.COLORS['OKBLUE']}[7/8] Создание скриптов запуска...{self.COLORS['ENDC']}")

        # Создаем bash скрипт для Linux/Mac
        bash_script = """#!/bin/bash
# Скрипт запуска AI Trader

echo "🤖 Запуск AI Trader..."
cd "$(dirname "$0")"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден"
    exit 1
fi

# Проверка зависимостей
if [ ! -f "requirements.txt" ]; then
    echo "❌ Файл requirements.txt не найден"
    exit 1
fi

# Запуск
python3 main.py
"""

        bash_path = self.project_root / "start.sh"
        if not bash_path.exists():
            with open(bash_path, 'w', encoding='utf-8') as f:
                f.write(bash_script)

            # Делаем исполняемым
            import stat
            bash_path.chmod(bash_path.stat().st_mode | stat.S_IEXEC)
            print(f"  🐧 Создан скрипт запуска: {bash_path.name}")

        # Создаем bat скрипт для Windows
        bat_script = """@echo off
REM Скрипт запуска AI Trader для Windows

echo 🤖 Запуск AI Trader...
cd /d "%~dp0"

REM Проверка Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python не найден
    pause
    exit /b 1
)

REM Запуск
python main.py
"""

        bat_path = self.project_root / "start.bat"
        if not bat_path.exists():
            with open(bat_path, 'w', encoding='utf-8') as f:
                f.write(bat_script)
            print(f"  🪟 Создан скрипт запуска: {bat_path.name}")

        print(f"{self.COLORS['OKGREEN']}✅ Скрипты запуска созданы{self.COLORS['ENDC']}")

    def run_tests(self):
        """Запуск тестов системы"""
        print(f"{self.COLORS['OKBLUE']}[8/8] Проверка системы...{self.COLORS['ENDC']}")

        tests_passed = 0
        tests_total = 5

        # Тест 1: Проверка Python файлов
        print("  🔍 Проверка Python файлов...", end="")
        try:
            import ast
            python_files = list(self.project_root.rglob("*.py"))
            for py_file in python_files[:10]:  # Проверяем первые 10 файлов
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            print(f"{self.COLORS['OKGREEN']} ✅{self.COLORS['ENDC']}")
            tests_passed += 1
        except SyntaxError as e:
            print(f"{self.COLORS['FAIL']} ❌ Ошибка синтаксиса в {e.filename}:{e.lineno}{self.COLORS['ENDC']}")

        # Тест 2: Проверка конфигурационных файлов
        print("  🔍 Проверка конфигурации...", end="")
        try:
            config_files = list(self.config_dir.glob("*.json"))
            for config_file in config_files:
                with open(config_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            print(f"{self.COLORS['OKGREEN']} ✅{self.COLORS['ENDC']}")
            tests_passed += 1
        except json.JSONDecodeError as e:
            print(f"{self.COLORS['FAIL']} ❌ Ошибка JSON в {e.doc}:{e.pos}{self.COLORS['ENDC']}")

        # Тест 3: Проверка импортов
        print("  🔍 Проверка импортов...", end="")
        try:
            import sys
            sys.path.insert(0, str(self.project_root))

            # Пробуем импортировать основные модули
            test_imports = [
                ("utils.logger", "setup_logger"),
                ("fetchers.moex_fetcher", "MoexFetcher"),
                ("models.smart_broker", "SmartPortfolioBroker")
            ]

            for module_name, class_name in test_imports:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)

            print(f"{self.COLORS['OKGREEN']} ✅{self.COLORS['ENDC']}")
            tests_passed += 1
        except ImportError as e:
            print(f"{self.COLORS['FAIL']} ❌ Ошибка импорта: {e}{self.COLORS['ENDC']}")

        # Тест 4: Проверка веб-шаблонов
        print("  🔍 Проверка веб-шаблонов...", end="")
        web_templates = list((self.project_root / "web" / "templates").glob("*.html"))
        if web_templates:
            print(f"{self.COLORS['OKGREEN']} ✅ ({len(web_templates)} файлов){self.COLORS['ENDC']}")
            tests_passed += 1
        else:
            print(f"{self.COLORS['WARNING']} ⚠ Нет HTML шаблонов{self.COLORS['ENDC']}")

        # Тест 5: Проверка статических файлов
        print("  🔍 Проверка статических файлов...", end="")
        static_files = list((self.project_root / "static").glob("*"))
        if static_files:
            print(f"{self.COLORS['OKGREEN']} ✅ ({len(static_files)} файлов){self.COLORS['ENDC']}")
            tests_passed += 1
        else:
            print(f"{self.COLORS['WARNING']} ⚠ Нет статических файлов{self.COLORS['ENDC']}")

        # Итог проверок
        print(f"\n📊 Результаты проверки: {tests_passed}/{tests_total}")

        if tests_passed == tests_total:
            print(f"{self.COLORS['OKGREEN']}✅ Все проверки пройдены успешно!{self.COLORS['ENDC']}")
        elif tests_passed >= tests_total * 0.7:
            print(
                f"{self.COLORS['WARNING']}⚠ Некоторые проверки не пройдены, но система должна работать{self.COLORS['ENDC']}")
        else:
            print(f"{self.COLORS['FAIL']}❌ Многие проверки не пройдены, возможны проблемы{self.COLORS['ENDC']}")

        return tests_passed >= tests_total * 0.7

    def show_completion_message(self):
        """Показать сообщение об успешной установке"""
        print(f"\n{self.COLORS['OKGREEN']}{'=' * 60}")
        print("🎉 УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
        print(f"{'=' * 60}{self.COLORS['ENDC']}")

        print("\n📋 Следующие шаги:")
        print(f"1. {self.COLORS['BOLD']}Настройте параметры в config/settings.json{self.COLORS['ENDC']}")
        print("2. Добавьте RSS источники в config/rss_sources.json")
        print("3. Настройте список тикеров в config/tickers.json")

        print(f"\n🚀 Запуск системы:")
        print(f"   Linux/Mac: {self.COLORS['OKCYAN']}./start.sh{self.COLORS['ENDC']}")
        print(f"   Windows: {self.COLORS['OKCYAN']}start.bat{self.COLORS['ENDC']}")
        print(f"   Или: {self.COLORS['OKCYAN']}python main.py{self.COLORS['ENDC']}")

        print(f"\n🌐 Веб-интерфейс будет доступен по адресу:")
        print(f"   {self.COLORS['OKCYAN']}http://localhost:8050{self.COLORS['ENDC']}")

        print(f"\n📖 Документация:")
        print(f"   {self.COLORS['OKCYAN']}README.md{self.COLORS['ENDC']} - Основная документация")
        print(f"   {self.COLORS['OKCYAN']}config/README_CONFIG.md{self.COLORS['ENDC']} - Настройка конфигурации")

        print(f"\n🆘 Поддержка:")
        print(f"   Проблемы с установкой: проверьте логи выше")
        print(f"   Вопросы по использованию: см. документацию")

        print(f"\n{self.COLORS['WARNING']}⚠ ВНИМАНИЕ:")
        print("   Алгоритмическая торговля связана с финансовыми рисками.")
        print("   Всегда тестируйте в режиме бумажной торговли перед использованием реальных средств.")
        print(f"{self.COLORS['ENDC']}")

    def run_installer(self):
        """Запуск процесса установки"""
        self.print_header()

        try:
            # Проверка версии Python
            if not self.check_python_version():
                return False

            # Создание директорий
            self.create_directories()

            # Установка зависимостей
            if not self.install_dependencies():
                print(f"{self.COLORS['WARNING']}⚠ Продолжаем без установки зависимостей...{self.COLORS['ENDC']}")

            # Создание конфигураций
            self.create_default_configs()

            # Создание начальных данных
            self.create_initial_data()

            # Настройка модели
            self.setup_model_weights()

            # Создание скриптов запуска
            self.create_startup_scripts()

            # Проверка системы
            self.run_tests()

            # Сообщение об успехе
            self.show_completion_message()

            return True

        except KeyboardInterrupt:
            print(f"\n{self.COLORS['WARNING']}⚠ Установка прервана пользователем{self.COLORS['ENDC']}")
            return False
        except Exception as e:
            print(f"\n{self.COLORS['FAIL']}❌ Критическая ошибка: {e}{self.COLORS['ENDC']}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Основная функция установщика"""
    installer = AITraderInstaller()

    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "help" or command == "--help" or command == "-h":
            print("Использование:")
            print("  python setup.py          - Полная установка")
            print("  python setup.py config   - Только конфигурация")
            print("  python setup.py test     - Только тестирование")
            return

        elif command == "config":
            installer.print_header()
            installer.create_directories()
            installer.create_default_configs()
            installer.create_initial_data()
            print(f"\n{self.COLORS['OKGREEN']}✅ Конфигурация создана{self.COLORS['ENDC']}")
            return

        elif command == "test":
            installer.print_header()
            success = installer.run_tests()
            sys.exit(0 if success else 1)

    # Полная установка
    success = installer.run_installer()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()