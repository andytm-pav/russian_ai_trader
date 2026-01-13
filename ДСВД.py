from datetime import datetime
import pytz
from trading_hours_scheduler import TradingScheduler

scheduler = TradingScheduler()

# Проверим дату с ДСВД (например, суббота, 17 января 2026)
test_date = datetime(2026, 1, 17, tzinfo=pytz.timezone('Europe/Moscow'))
print(f"17.01.2026 (суббота, ДСВД): is_trading_day = {scheduler.is_trading_day(test_date)}")
print(f"                                is_trading_time в 10:00 = {scheduler.is_trading_time(test_date.replace(hour=10))}")

# Проверим дату без ДСВД (суббота, 10 января 2026)
test_date_excluded = datetime(2026, 1, 10, tzinfo=pytz.timezone('Europe/Moscow'))
print(f"10.01.2026 (суббота, исключение): is_trading_day = {scheduler.is_trading_day(test_date_excluded)}")