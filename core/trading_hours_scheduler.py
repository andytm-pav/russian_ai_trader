"""
Планировщик торговых сессий - контроль времени торговли
"""

import json
import pytz
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import schedule
import time as ttime

from utils.logger import setup_logger

logger = setup_logger("SCHEDULER")


class TradingScheduler:
    """Управление торговыми сессиями и временем"""

    def __init__(self, config_path: str = "config/market_schedule.json"):
        self.config = self._load_config(config_path)
        self.moscow_tz = pytz.timezone('Europe/Moscow')
        self.today_cache = None
        self.is_trading_day_cache = None

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
        """Проверка, является ли день торговым"""
        if check_date is None:
            check_date = datetime.now(self.moscow_tz)

        # Проверка кэша
        date_str = check_date.strftime('%Y-%m-%d')
        if self.today_cache == date_str and self.is_trading_day_cache is not None:
            return self.is_trading_day_cache

        date_str_short = check_date.strftime('%Y-%m-%d')
        day_of_week = check_date.strftime('%A').lower()

        # Проверка регулярных дней и выходных сессий (ДСВД)
        is_regular_weekday = day_of_week in self.config['trading_calendar']['regular_days']
        is_weekend = day_of_week in ['saturday', 'sunday']

        if not is_regular_weekday:
            # Для субботы и воскресенья проверяем, не входят ли они в список исключений для ДСВД
            if is_weekend:
                excluded_dates = self.config.get('weekend_sessions', {}).get('excluded_dates_2026', [])
                if date_str_short in excluded_dates:
                    self._update_cache(date_str, False)
                    return False
                else:
                    # Это выходной, но на него назначена ДСВД
                    self._update_cache(date_str, True)
                    return True
            # Если это не будний день и не выходной с ДСВД (например, праздник в середине недели)
            self._update_cache(date_str, False)
            return False

        # Проверка праздников
        date_str_short = check_date.strftime('%Y-%m-%d')
        if date_str_short in self.config.get('holidays_2026', []):
            self._update_cache(date_str, False)
            return False

        # Проверка специальных торговых дней
        if date_str_short in self.config.get('special_trading_days', []):
            self._update_cache(date_str, True)
            return True

        # Проверка предпраздничных дней
        if date_str_short in self.config.get('trading_calendar', {}).get('pre_holiday_early_close', []):
            self._update_cache(date_str, True)
            return True

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

    def get_next_session_start(self) -> Optional[datetime]:
        """Получение времени начала следующей сессии"""
        now = datetime.now(self.moscow_tz)

        # Проверяем на 7 дней вперед
        for days_ahead in range(8):
            check_date = now + timedelta(days=days_ahead)

            if self.is_trading_day(check_date):
                # Получаем время начала первой сессии дня
                sessions = self.config.get('sessions', {})

                # Главная сессия
                main_session = sessions.get('main_session', {})
                if main_session.get('enabled', True):
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

    def schedule_daily_tasks(self,
                             pre_market_callback,
                             market_open_callback,
                             market_close_callback,
                             post_market_callback):
        """Планирование ежедневных задач"""

        # Предрыночный анализ (06:30)
        schedule.every().day.at("06:30").do(pre_market_callback)
        logger.info("Запланирован предрыночный анализ на 06:30")

        # Открытие рынка (06:50)
        schedule.every().day.at("06:50").do(market_open_callback)
        logger.info("Запланировано открытие рынка на 06:50")

        # Закрытие рынка (18:50)
        schedule.every().day.at("18:50").do(market_close_callback)
        logger.info("Запланировано закрытие рынка на 18:50")

        # Послерыночный анализ (19:00)
        schedule.every().day.at("19:00").do(post_market_callback)
        logger.info("Запланирован послерыночный анализ на 19:00")

        # Запуск планировщика в отдельном потоке
        import threading

        def schedule_runner():
            while True:
                schedule.run_pending()
                ttime.sleep(60)

        scheduler_thread = threading.Thread(target=schedule_runner, daemon=True)
        scheduler_thread.start()
        logger.info("Планировщик задач запущен")