"""
Планировщик торговых сессий - контроль времени торговли
"""

import json
import pytz
import threading  # ✅ ДОБАВЛЕНО
import time as ttime
import schedule
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("SCHEDULER")


class TradingScheduler:
    """Управление торговыми сессиями и временем"""

    def __init__(self, config_path: str = "config/market_schedule.json"):
        self.config = self._load_config(config_path)
        self.moscow_tz = pytz.timezone('Europe/Moscow')
        self.today_cache = None
        self.is_trading_day_cache = None
        # ✅ ДОБАВЛЕНО
        self.scheduler_thread = None
        self.scheduler_running = False

        logger.info("Инициализирован планировщик торговых сессий")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации торговых сессий"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации сессий: {e}")
            return {
                "trading_calendar": {
                    "regular_days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
                },
                "sessions": {
                    "main_session": {
                        "start": "06:50",
                        "end": "18:50"
                    }
                },
                "holidays_2026": []
            }

    def is_trading_day(self, check_date: Optional[datetime] = None) -> bool:
        """Проверка, является ли день торговым (с обратной совместимостью)"""
        if check_date is None:
            check_date = datetime.now(self.moscow_tz)

        date_str = check_date.strftime('%Y-%m-%d')

        # --- СБРОС КЭША  ---
        if self.today_cache is not None and self.today_cache != date_str:
            # logger.debug(f"is_trading_day: смена даты {self.today_cache} -> {date_str}")
            self.today_cache = None
            self.is_trading_day_cache = None

        # Проверка кэша
        if self.today_cache == date_str and self.is_trading_day_cache is not None:
            return self.is_trading_day_cache

        day_of_week = check_date.strftime('%A').lower()

        # БЕЗОПАСНОЕ получение конфигов с default-значениями
        trading_calendar = self.config.get('trading_calendar', {})
        regular_days = trading_calendar.get('regular_days', ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'])

        # Получаем holidays с динамическим годом
        current_year = check_date.strftime('%Y')
        holidays_key = f'holidays_{current_year}'
        holidays = self.config.get(holidays_key, self.config.get('holidays_2026', []))  # fallback на 2026

        # Получаем excluded_dates с динамическим годом
        weekend_sessions = self.config.get('weekend_sessions', {})
        excluded_key = f'excluded_dates_{current_year}'
        excluded_dates = weekend_sessions.get(excluded_key, weekend_sessions.get('excluded_dates_2026', []))

        # Специальные торговые дни
        special_days = self.config.get('special_trading_days', [])

        # Предпраздничные дни
        pre_holiday = trading_calendar.get('pre_holiday_early_close', [])

        # Определяем тип дня
        is_regular_weekday = day_of_week in regular_days
        is_weekend = day_of_week in ['saturday', 'sunday']

        # 1. ПРАЗДНИКИ - всегда нет торгов
        if date_str in holidays:
            self._update_cache(date_str, False)
            return False

        # 2. СПЕЦИАЛЬНЫЕ ТОРГОВЫЕ ДНИ - всегда есть
        if date_str in special_days:
            self._update_cache(date_str, True)
            return True

        # 3. ОБРАБОТКА ВЫХОДНЫХ
        if not is_regular_weekday:
            if is_weekend:
                # Проверяем, включена ли ДСВД
                if weekend_sessions.get('enabled', False):
                    # Проверяем исключения
                    if date_str in excluded_dates:
                        self._update_cache(date_str, False)
                        return False
                    else:
                        self._update_cache(date_str, True)
                        return True
                else:
                    # ДСВД отключена глобально
                    self._update_cache(date_str, False)
                    return False
            # Не будний и не выходной (например, праздник среди недели)
            self._update_cache(date_str, False)
            return False

        # 4. БУДНИЙ ДЕНЬ - проверяем предпраздничные
        if date_str in pre_holiday:
            self._update_cache(date_str, True)
            return True

        # 5. ОБЫЧНЫЙ БУДНИЙ ДЕНЬ
        self._update_cache(date_str, True)
        return True

    def _update_cache(self, date_str: str, is_trading: bool):
        """Обновление кэша"""
        self.today_cache = date_str
        self.is_trading_day_cache = is_trading

    def is_trading_time(self, check_datetime: Optional[datetime] = None) -> bool:
        """Проверка, идет ли сейчас торговая сессия"""
        if check_datetime is None:
            check_datetime = datetime.now(self.moscow_tz)

        # --- Принудительный сброс кэша при смене дня ---
        #current_date_str = check_datetime.strftime('%Y-%m-%d')
        #if self.today_cache is not None and self.today_cache != current_date_str:
        #    # Дата изменилась, сбрасываем кэш
        #    logger.debug(f"Смена даты: {self.today_cache} -> {current_date_str}. Сброс кэша.")
        #    self.today_cache = None
        #    self.is_trading_day_cache = None

        # Проверка дня
        if not self.is_trading_day(check_datetime):
            return False

        # Получение времени
        current_time = check_datetime.time()

        # Проверка основных сессий
        sessions = self.config.get('sessions', {})

        # Главная сессия
        main_session = sessions.get('main_session', {})
        if main_session.get('enabled', True):
            start_time = self._parse_time(main_session.get('start', '06:50'))
            end_time = self._parse_time(main_session.get('end', '18:50'))

            if start_time <= current_time <= end_time:
                return True

        # Вечерняя сессия
        evening_session = sessions.get('evening_session', {})
        if evening_session.get('enabled', False):
            start_time = self._parse_time(evening_session.get('start', '19:00'))
            end_time = self._parse_time(evening_session.get('end', '23:50'))

            if start_time <= current_time <= end_time:
                return True

        # Ночная сессия
        overnight_session = sessions.get('overnight_session', {})
        if overnight_session.get('enabled', False):
            start_time = self._parse_time(overnight_session.get('start', '00:00'))
            end_time = self._parse_time(overnight_session.get('end', '06:50'))

            if start_time <= current_time <= end_time:
                return True

        # Сессия в выходной день (ДСВД)
        weekend_session = sessions.get('weekend_session', {})
        if weekend_session.get('enabled', False):
            # ⚠️ ПРОВЕРКА ИСКЛЮЧЕНИЙ
            date_str = check_datetime.strftime('%Y-%m-%d')
            weekend_sessions_config = self.config.get('weekend_sessions', {})
            current_year = check_datetime.strftime('%Y')
            excluded_key = f'excluded_dates_{current_year}'
            excluded_dates = weekend_sessions_config.get(excluded_key,
                     weekend_sessions_config.get('excluded_dates_2026', []))

            if date_str in excluded_dates:
                return False

             # ✅ ДОБАВИТЬ ПРОВЕРКУ enabled ИЗ weekend_sessions_config
            if not weekend_sessions_config.get('enabled', False):
                return False  # ДСВД отключена глобально

            # Получаем день недели из текущей даты
            day_of_week = check_datetime.strftime('%A').lower()

            if day_of_week in ['saturday', 'sunday']:
                start_time = self._parse_time(weekend_session.get('start', '09:50'))
                end_time = self._parse_time(weekend_session.get('end', '19:00'))

                if start_time <= current_time <= end_time:
                    return True

        # Проверка обеденного перерыва
        market_breaks = self.config.get('market_breaks', {})
        lunch_break = market_breaks.get('lunch_break', {})

        if lunch_break.get('enabled', False):
            break_start = self._parse_time(lunch_break.get('start', '12:00'))
            break_end = self._parse_time(lunch_break.get('end', '12:30'))

            if break_start <= current_time <= break_end:
                return False

        return False

    def _parse_time(self, time_str: str) -> time:
        """Парсинг строки времени"""
        try:
            return datetime.strptime(time_str, '%H:%M').time()
        except:
            return datetime.strptime('09:00', '%H:%M').time()

    # ✅ НОВЫЙ МЕТОД
    def _time_in_range(self, start: str, end: str, current: time) -> bool:
        """Проверка, находится ли текущее время в диапазоне"""
        start_time = self._parse_time(start)
        end_time = self._parse_time(end)

        if start_time <= end_time:
            return start_time <= current <= end_time
        else:  # Диапазон переходит через полночь
            return current >= start_time or current <= end_time

    # ✅ НОВЫЙ МЕТОД
    def get_current_moex_period(self) -> str:
        """Определение текущего периода торгов MOEX"""
        if not self.is_trading_time():
            return "closed"

        now = datetime.now(self.moscow_tz).time()

        # Используем settings.json для получения периодов
        try:
            with open("config/settings.json", "r") as f:
                settings = json.load(f)
                moex = settings.get("moex_schedule", {})
                periods = moex.get("periods", {})
        except:
            # Fallback - стандартные периоды MOEX
            periods = {
                "auction_open": {"start": "09:50", "end": "09:59"},
                "continuous_trading": {"start": "10:00", "end": "18:40"},
                "auction_close": {"start": "18:40", "end": "18:50"},
                "evening_auction_open": {"start": "19:00", "end": "19:05"},
                "evening_continuous": {"start": "19:05", "end": "23:50"}
            }

        for period_name, times in periods.items():
            if self._time_in_range(times["start"], times["end"], now):
                return period_name
        return "continuous_trading"

    # ✅ НОВЫЙ МЕТОД
    def can_trade_now(self) -> Dict[str, bool]:
        """Проверка доступности торговых операций"""
        period = self.get_current_moex_period()


        return {
            'can_place_orders': period in ['auction_open', 'continuous_trading', 'auction_close', 'evening_auction_open', 'evening_continuous'],
            'can_cancel_orders': period in ['auction_open', 'continuous_trading', 'auction_close','evening_auction_open', 'evening_continuous'],
            'can_modify_orders': period in ['continuous_trading', 'evening_continuous'],
            'current_period': period
        }

    def get_next_session_start(self) -> Optional[datetime]:
        """Получение времени начала следующей сессии"""
        now = datetime.now(self.moscow_tz)

        # Проверяем на 7 дней вперед
        for days_ahead in range(8):
            check_date = now + timedelta(days=days_ahead)

            if self.is_trading_day(check_date):
                # Получаем время начала сессии в зависимости от дня недели
                day_of_week = check_date.strftime('%A').lower()
                sessions = self.config.get('sessions', {})

                # Для выходных дней используем ДСВД
                if day_of_week in ['saturday', 'sunday']:
                    weekend_session = sessions.get('weekend_session', {})
                    if weekend_session.get('enabled', True):
                        start_time = self._parse_time(weekend_session.get('start', '09:50'))
                else:
                    # Для будних дней используем основную сессию
                    main_session = sessions.get('main_session', {})
                    start_time = self._parse_time(main_session.get('start', '06:50'))

                session_start = datetime.combine(
                    check_date.date(),
                    start_time,
                    self.moscow_tz
                )

                # Если это сегодня и время уже прошло, ищем завтра
                if days_ahead == 0 and session_start <= now:
                    continue

                return session_start

        return None

    def get_time_to_next_session(self) -> Optional[Tuple[int, int, int]]:
        """Время до следующей сессии (часы, минуты, секунды)"""
        next_session = self.get_next_session_start()

        if next_session:
            now = datetime.now(self.moscow_tz)
            delta = next_session - now

            if delta.total_seconds() > 0:
                hours = delta.seconds // 3600
                minutes = (delta.seconds % 3600) // 60
                seconds = delta.seconds % 60
                return hours, minutes, seconds

        return None

    def get_session_info(self) -> Dict:
        """Получение информации о текущей сессии"""
        now = datetime.now(self.moscow_tz)

        info = {
            'current_time': now.isoformat(),
            'is_trading_day': self.is_trading_day(now),
            'is_trading_time': self.is_trading_time(now),
            'day_of_week': now.strftime('%A'),
            'date': now.strftime('%Y-%m-%d')
        }

        # Добавляем информацию о сессии
        if info['is_trading_time']:
            current_time = now.time()
            sessions = self.config.get('sessions', {})

            # Определяем, какая сессия активна
            for session_name, session_data in sessions.items():
                if session_data.get('enabled', True):
                    start_time = self._parse_time(session_data.get('start', '00:00'))
                    end_time = self._parse_time(session_data.get('end', '23:59'))

                    if start_time <= current_time <= end_time:
                        info['active_session'] = session_name
                        info['session_start'] = start_time.strftime('%H:%M')
                        info['session_end'] = end_time.strftime('%H:%M')

                        # Время до конца сессии
                        session_end_dt = datetime.combine(
                            now.date(),
                            end_time,
                            self.moscow_tz
                        )
                        time_left = session_end_dt - now

                        if time_left.total_seconds() > 0:
                            info['minutes_to_close'] = int(time_left.total_seconds() // 60)
                        break

        # Время до следующей сессии
        next_session = self.get_next_session_start()
        if next_session:
            info['next_session_start'] = next_session.isoformat()

            time_to_next = self.get_time_to_next_session()
            if time_to_next:
                info['hours_to_next'] = time_to_next[0]
                info['minutes_to_next'] = time_to_next[1]

        return info

    # ============================================
    # ✅ НОВЫЕ МЕТОДЫ ДЛЯ ПЛАНИРОВЩИКА
    # ============================================

    def _execute_with_context(self, callback, context: str):
        """Выполнение callback с обработкой ошибок"""
        try:
            logger.info(f"🔔 ВЫПОЛНЕНИЕ: {context}")
            result = callback()
            logger.info(f"✅ Завершено: {context} -> {result}")
            return result
        except Exception as e:
            logger.error(f"❌ Ошибка в {context}: {e}")
            return False

    def _log_clearing_fixation(self):
        """16:00 - Фиксация обязательств Т+"""
        logger.info("🔔 MOEX: 16:00 - Фиксация обязательств Т+")
        return True

    def _log_z0_deadline(self):
        """16:50 - Дедлайн сделок Z0"""
        logger.info("🔔 MOEX: 16:50 - Дедлайн Z0 (РПС/РЕПО)")
        return True

    def _log_commission_charge(self):
        """19:00 - Списание комиссий"""
        logger.info("🔔 MOEX: 19:00 - Списание комиссий Т+0")
        return True

    def _start_scheduler(self):
        """Запуск планировщика в отдельном потоке"""
        if self.scheduler_running:
            return

        def schedule_runner():
            self.scheduler_running = True
            while self.scheduler_running:
                try:
                    schedule.run_pending()
                    ttime.sleep(1)
                except Exception as e:
                    logger.error(f"Ошибка в планировщике: {e}")
                    ttime.sleep(5)

        self.scheduler_thread = threading.Thread(target=schedule_runner, daemon=True)
        self.scheduler_thread.start()
        logger.info("Планировщик задач запущен")

    def stop_scheduler(self):
        """Остановка планировщика"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Планировщик задач остановлен")

    # ✅ ИСПРАВЛЕННЫЙ МЕТОД schedule_daily_tasks
    def schedule_daily_tasks(self,
                             pre_market_callback,
                             market_open_callback,
                             market_close_callback,
                             post_market_callback,
                             clearing_liquidity_callback=None,
                             z0_deadline_callback=None,
                             clearing_17_callback=None,
                             clearing_19_callback=None,
                             commission_callback=None):
        """Планирование ежедневных задач с поддержкой вечерней сессии"""

        # Читаем ВСЕ времена из конфига - никаких магических чисел!
        try:
            with open("config/settings.json", "r") as f:
                settings = json.load(f)
                moex = settings.get("moex_schedule", {})
                periods = moex.get("periods", {})
                clearing = moex.get("clearing", {})
                deadlines = clearing.get("deadlines", {})
                commission = moex.get("commission", {})

                # Утренние задачи
                pre_market = moex.get("pre_market_start", "06:30")
                market_open = periods.get("continuous_trading", {}).get("start", "10:00")

                # Вечернее закрытие - когда реально прекращаются торги (23:50)
                evening_close = periods.get("evening_continuous", {}).get("end", "23:50")

                # Послерыночный анализ ПОСЛЕ закрытия вечерней сессии
                post_market = moex.get("post_market_start", "23:55")

                # Бэкофис задачи
                fixation = clearing.get("fixation_time", "16:00")
                z0_cutoff = deadlines.get("z0", "16:50")
                clearing_17_time = clearing.get("clearing_17", "17:00")
                clearing_19_time = clearing.get("clearing_19", "19:00")
                charge_time = commission.get("charge_time", "19:00")

        except Exception as e:
            # ПОЛНАЯ ОБРАТНАЯ СОВМЕСТИМОСТЬ - fallback значения
            logger.warning(f"Не удалось загрузить конфиг, использую fallback: {e}")
            pre_market = "06:30"
            market_open = "10:00"
            evening_close = "23:50"
            post_market = "23:55"
            fixation = "16:00"
            z0_cutoff = "16:50"
            clearing_17_time = "17:00"
            clearing_19_time = "19:00"
            charge_time = "19:00"

        # Очищаем существующие задачи
        schedule.clear()

        # === ОСНОВНЫЕ ЗАДАЧИ ===
        # Предрыночный анализ (всегда выполняется)
        schedule.every().day.at(pre_market).do(pre_market_callback)
        logger.info(f"✅ Запланирован предрыночный анализ на {pre_market}")

        # Открытие рынка
        schedule.every().day.at(market_open).do(market_open_callback)
        logger.info(f"✅ Запланировано открытие рынка на {market_open}")

        # ⚠️ Закрытие рынка ТОЛЬКО после вечерней сессии!
        schedule.every().day.at(evening_close).do(market_close_callback)
        logger.info(f"✅ Запланировано закрытие рынка на {evening_close}")

        # Послерыночный анализ после закрытия
        schedule.every().day.at(post_market).do(post_market_callback)
        logger.info(f"✅ Запланирован послерыночный анализ на {post_market}")

        # === БЭКОФИС ЗАДАЧИ (только если переданы callback'и) ===
        if clearing_liquidity_callback:
            schedule.every().day.at(fixation).do(
                lambda: self._execute_with_context(clearing_liquidity_callback, "fixation")
            )
            logger.info(f"✅ Запланирована проверка ликвидности на {fixation}")

        if z0_deadline_callback:
            schedule.every().day.at(z0_cutoff).do(
                lambda: self._execute_with_context(z0_deadline_callback, "z0")
            )
            logger.info(f"✅ Запланирован дедлайн Z0 на {z0_cutoff}")

        if clearing_17_callback:
            schedule.every().day.at(clearing_17_time).do(
                lambda: self._execute_with_context(clearing_17_callback, "clearing_17")
            )
            logger.info(f"✅ Запланирован клиринг 17:00 на {clearing_17_time}")

        if clearing_19_callback:
            schedule.every().day.at(clearing_19_time).do(
                lambda: self._execute_with_context(clearing_19_callback, "clearing_19")
            )
            logger.info(f"✅ Запланирован клиринг 19:00 на {clearing_19_time}")

        # === СПИСАНИЕ КОМИССИЙ ===
        if commission_callback:
            schedule.every().day.at("14:00").do(
                lambda: self._execute_with_context(commission_callback, "commission_charge")
            )
            logger.info(f"✅ Запланировано списание комиссий на 14:00 (T+1)")

        # === ЛОГИРОВАНИЕ (всегда активно) ===
        schedule.every().day.at(fixation).do(self._log_clearing_fixation)
        schedule.every().day.at(z0_cutoff).do(self._log_z0_deadline)
        schedule.every().day.at(charge_time).do(self._log_commission_charge)

        # Запуск планировщика
        self._start_scheduler()