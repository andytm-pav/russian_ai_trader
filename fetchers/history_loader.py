#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Загрузчик исторических данных MOEX.

При первом старте:
  - Загружает 3 месяца часовых свечей для всех тикеров TQBR
  - Кормит их в technical_core.price_history
  - Рассчитывает и сохраняет хаос-метрики (Херст, Ляпунов, D₂)

При последующих стартах:
  - Проверяет последнюю дату в price_history.json
  - Дозагружает только недостающие данные
  - Пересчитывает хаос-метрики на обновлённом окне

Также загружает:
  - Часовые свечи IMOEX, RTSI, MOEXOG, MOEXFN, MOEXMM
  - Дневные курсы USDRUB, CNYRUB
"""
import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger("HISTORY_LOADER")


class HistoryLoader:
    """Загрузчик исторических данных с инкрементальным обновлением."""

    HISTORY_FILE = "data/price_history_extended.json"
    CHAOS_CACHE_FILE = "data/chaos_metrics_cache.json"

    # Тикеры для обязательной загрузки (индексы + макро)
    INDEX_TICKERS = ["IMOEX", "MOEXOG", "MOEXFN", "MOEXMM", "MOEXCN", "MOEXEU",
                     "MOEXIT", "MOEXRE", "MOEXTL", "MOEXCH"]

    def __init__(self, moex_fetcher=None, technical_core=None):
        self.moex = moex_fetcher
        self.tech_core = technical_core
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://iss.moex.com/iss"

        # Загрузка конфигурации
        self._load_config()

        logger.info(f"HistoryLoader инициализирован (months_back={self.months_back}, max_points={self.max_points})")

    def _load_config(self):
        """Загрузка параметров из config/settings.json"""
        import json
        self.history_file = "data/price_history_extended.json"
        self.chaos_cache_file = "data/chaos_metrics_cache.json"
        self.months_back = 3
        self.max_points = 2000
        self.rate_limit = 0.15
        self.top_tickers_limit = 50

        try:
            with open("config/settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
            hl_config = settings.get("history_loader", {})
            self.months_back = hl_config.get("months_back", 3)
            self.max_points = hl_config.get("max_points_per_ticker", 2000)
            self.rate_limit = hl_config.get("rate_limit_seconds", 0.15)
            self.top_tickers_limit = hl_config.get("top_tickers_limit", 50)
            self.history_file = hl_config.get("history_file", self.history_file)
            self.chaos_cache_file = hl_config.get("chaos_cache_file", self.chaos_cache_file)
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфиг history_loader: {e}, использую значения по умолчанию")

    def load_history(self, tickers: List[str] = None,
                     months_back: int = None) -> Dict[str, Dict]:
        """
        Загрузка исторических данных.
        
        При первом старте: загружает months_back месяцев (из конфига).
        При последующих: дозагружает только недостающие.
        
        Возвращает словарь {ticker: {prices: [], volumes: [], timestamps: []}}
        """
        if months_back is None:
            months_back = self.months_back

        # Загружаем существующую историю
        existing = self._load_existing_history()
        
        # Определяем диапазон загрузки
        now = datetime.now()
        default_start = now - timedelta(days=30 * months_back)
        
        if not existing:
            logger.info(f"=== ПЕРВЫЙ СТАРТ: загрузка {months_back} мес истории ===")
            start_date = default_start
        else:
            # Находим самую свежую дату в истории
            latest_date = self._get_latest_date(existing)
            if latest_date:
                # Дозагружаем с последней даты + 1 час
                start_date = latest_date + timedelta(hours=1)
                hours_gap = (now - start_date).total_seconds() / 3600
                if hours_gap < 1:
                    logger.info(f"История актуальна (последняя точка: {latest_date}), пропускаем загрузку")
                    self._feed_to_technical_core(existing)
                    # 🆕 v14.9: Хаос-метрики нужно пересчитать даже при актуальной истории
                    # (если кэш удалён или устарел)
                    if not os.path.exists(self.chaos_cache_file) or self._is_chaos_cache_stale():
                        self._calculate_chaos_metrics(existing)
                    return existing
                logger.info(f"=== ИНКРЕМЕНТ: дозагрузка с {start_date} ({hours_gap:.0f}ч не хватает) ===")
            else:
                start_date = default_start

        # Определяем тикеры для загрузки
        if tickers is None:
            tickers = self._get_top_tickers()

        # Добавляем индексы
        all_tickers = list(set(tickers + self.INDEX_TICKERS))

        logger.info(f"Тикеров к загрузке: {len(all_tickers)}")
        logger.info(f"Период: {start_date.strftime('%Y-%m-%d')} → {now.strftime('%Y-%m-%d')}")

        # Загружаем часовые свечи для каждого тикера
        new_data = {}
        for i, ticker in enumerate(all_tickers):
            try:
                candles = self._download_hourly_candles(ticker, start_date, now)
                if candles:
                    new_data[ticker] = candles
                    if (i + 1) % 20 == 0:
                        logger.info(f"  Загружено {i+1}/{len(all_tickers)} тикеров")
                time.sleep(0.15)  # Rate limit
            except Exception as e:
                logger.debug(f"Ошибка загрузки {ticker}: {e}")

        logger.info(f"Загружено данных для {len(new_data)} тикеров")

        # Объединяем с существующей историей
        merged = self._merge_history(existing, new_data)
        
        # Сохраняем
        self._save_history(merged)
        
        # Кормим в technical_core
        self._feed_to_technical_core(merged)
        
        # Считаем хаос-метрики
        self._calculate_chaos_metrics(merged)
        
        return merged

    def _load_existing_history(self) -> Dict:
        """Загрузка существующей истории из файла."""
        # Сначала пробуем extended файл
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Загружена существующая история: {len(data)} тикеров")
                return data
            except Exception as e:
                logger.warning(f"Ошибка загрузки истории: {e}")
        
        # Пробуем стандартный файл technical_core
        if self.tech_core and os.path.exists(self.tech_core.price_history_file):
            try:
                with open(self.tech_core.price_history_file, 'r') as f:
                    data = json.load(f)
                # Конвертируем формат
                converted = {}
                for ticker, hist in data.items():
                    converted[ticker] = {
                        'prices': hist.get('prices', []),
                        'volumes': hist.get('volumes', []),
                        'timestamps': hist.get('timestamps', []),
                    }
                if converted:
                    logger.info(f"Загружена история из technical_core: {len(converted)} тикеров")
                return converted
            except Exception:
                pass
        
        return {}

    def _get_latest_date(self, history: Dict) -> Optional[datetime]:
        """Находит самую свежую дату в истории."""
        latest = None
        for ticker, hist in history.items():
            timestamps = hist.get('timestamps', [])
            if timestamps:
                try:
                    ts = timestamps[-1]
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts)
                    else:
                        dt = datetime.fromisoformat(str(ts))
                    if latest is None or dt > latest:
                        latest = dt
                except Exception:
                    continue
        return latest

    def _get_top_tickers(self, limit: int = None) -> List[str]:
        """Получение топ-тикеров по ликвидности."""
        if limit is None:
            limit = self.top_tickers_limit
        if self.moex:
            try:
                securities = self.moex.get_all_securities()
                top = sorted(securities.items(), key=lambda x: x[1].get('volume', 0), reverse=True)
                return [t[0] for t in top[:limit] if t[1].get('price', 0) > 0]
            except Exception:
                pass
        # Fallback
        return ['SBER', 'GAZP', 'LKOH', 'MOEX', 'GMKN', 'NVTK', 'ROSN', 'TATN',
                'SNGS', 'MTSS', 'CHMF', 'MAGN', 'PLZL', 'AFKS', 'OZON']

    def _download_hourly_candles(self, ticker: str, start: datetime,
                                  end: datetime) -> Optional[Dict]:
        """Скачивание часовых свечей для тикера (close, high, low, open, volume)."""
        # Для индексов — другой URL
        if ticker in self.INDEX_TICKERS:
            url = f"{self.base_url}/engines/stock/markets/index/securities/{ticker}/candles.json"
        else:
            url = f"{self.base_url}/engines/stock/markets/shares/securities/{ticker}/candles.json"

        all_prices = []
        all_highs = []
        all_lows = []
        all_opens = []
        all_volumes = []
        all_timestamps = []

        cur = start
        while cur < end:
            next_batch = min(cur + timedelta(days=60), end)
            params = {
                'interval': 60,  # часовые
                'from': cur.strftime('%Y-%m-%d'),
                'till': next_batch.strftime('%Y-%m-%d'),
                'iss.meta': 'off',
                'iss.only': 'candles',
                'start': 0,
            }
            try:
                r = self.session.get(url, params=params, timeout=15)
                if r.status_code != 200:
                    break
                data = r.json()
                cols = data.get("candles", {}).get("columns", [])
                rows = data.get("candles", {}).get("data", [])
                if not rows:
                    break
                for row in rows:
                    d = dict(zip(cols, row))
                    close = d.get('close')
                    if close and close > 0:
                        all_prices.append(float(close))
                        all_highs.append(float(d.get('high', close)))
                        all_lows.append(float(d.get('low', close)))
                        all_opens.append(float(d.get('open', close)))
                        all_volumes.append(float(d.get('volume', 0)))
                        ts = d.get('begin', '')
                        all_timestamps.append(ts)
                if len(rows) < 500:
                    break
                params['start'] += 500
            except Exception:
                break
            cur = next_batch + timedelta(days=1)
            time.sleep(self.rate_limit)  # Из конфига

        if not all_prices:
            return None

        return {
            'prices': all_prices,
            'highs': all_highs,
            'lows': all_lows,
            'opens': all_opens,
            'volumes': all_volumes,
            'timestamps': all_timestamps,
        }

    def _download_intraday_candles(self, ticker: str,
                                    interval_minutes: int = 10,
                                    days_back: int = 5) -> Optional[Dict]:
        """
        🆕 Загрузка внутридневных свечей для bootstrap Hawkes.

        Использует более короткий интервал (10 минут по умолчанию), чтобы получить
        больше событий |log-return| > threshold. Это критически важно, потому что
        при часовых свечах и пороге ~0.5% событий слишком мало (<10) и Hawkes
        не может обучиться.

        Args:
            ticker: тикер
            interval_minutes: 1, 10, 60 (минуты)
            days_back: сколько дней назад тянуть данные

        Returns:
            {prices: [], timestamps: [], volumes: []} или None
        """
        # Для индексов — другой URL
        if ticker in self.INDEX_TICKERS:
            url = f"{self.base_url}/engines/stock/markets/index/securities/{ticker}/candles.json"
        else:
            url = f"{self.base_url}/engines/stock/markets/shares/securities/{ticker}/candles.json"

        end = datetime.now()
        start = end - timedelta(days=days_back)

        all_prices = []
        all_volumes = []
        all_timestamps = []

        cur = start
        while cur < end:
            next_batch = min(cur + timedelta(days=30), end)
            params = {
                'interval': interval_minutes,
                'from': cur.strftime('%Y-%m-%d'),
                'till': next_batch.strftime('%Y-%m-%d'),
                'iss.meta': 'off',
                'iss.only': 'candles',
                'start': 0,
            }
            try:
                r = self.session.get(url, params=params, timeout=15)
                if r.status_code != 200:
                    break
                data = r.json()
                cols = data.get("candles", {}).get("columns", [])
                rows = data.get("candles", {}).get("data", [])
                if not rows:
                    break
                for row in rows:
                    d = dict(zip(cols, row))
                    close = d.get('close')
                    if close and close > 0:
                        all_prices.append(float(close))
                        all_volumes.append(float(d.get('volume', 0)))
                        ts_str = d.get('begin', '')
                        # Парсим ISO timestamp
                        try:
                            if isinstance(ts_str, str):
                                dt_obj = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                                all_timestamps.append(dt_obj.timestamp())
                            else:
                                all_timestamps.append(float(ts_str))
                        except Exception:
                            all_timestamps.append(0.0)
                if len(rows) < 500:
                    break
                params['start'] += 500
            except Exception as e:
                logger.debug(f"Intraday {ticker}: ошибка batch {cur}: {e}")
                break
            cur = next_batch + timedelta(days=1)
            time.sleep(self.rate_limit)

        if not all_prices:
            return None

        logger.debug(f"Intraday {ticker}: {len(all_prices)} свечей ({interval_minutes}м × {days_back}д)")
        return {
            'prices': all_prices,
            'volumes': all_volumes,
            'timestamps': all_timestamps,
        }

    def bootstrap_hawkes_from_intraday(self, hawkes_instance,
                                        tickers: List[str] = None,
                                        interval_minutes: int = 10,
                                        days_back: int = 5,
                                        chaos_metrics: Dict = None) -> Dict:
        """
        🆕 Bootstrap Hawkes-процесса из внутридневных свечей MOEX.

        Загружает 10-минутные свечи за последние 5 дней (~720 свечей на тикер)
        и фидит их в hawkes.update_price() с правильными timestamp'ами.
        Затем вызывает hawkes.fit() для каждого тикера.

        Args:
            hawkes_instance: экземпляр HawkesSignalGenerator
            tickers: список тикеров (если None — берём топ из MOEX)
            interval_minutes: интервал свечей (по умолчанию 10 минут)
            days_back: глубина истории (по умолчанию 5 дней)
            chaos_metrics: dict {ticker: {volatility_pct: ...}} для калибровки порогов

        Returns:
            {ticker: {events_bullish: N, events_bearish: M, fitted: bool, eta_bull: float, ...}}
        """
        if hawkes_instance is None:
            logger.warning("Hawkes bootstrap: hawkes_instance is None, пропускаю")
            return {}

        if tickers is None:
            tickers = self._get_top_tickers(limit=30)

        # Обязательно добавляем индексы
        all_tickers = list(set(tickers + ["IMOEX"]))

        logger.info(f"🎯 Hawkes bootstrap: {len(all_tickers)} тикеров, "
                    f"interval={interval_minutes}м, days_back={days_back}")

        # 🆕 Сначала калибруем пороги по волатильности, если есть chaos_metrics
        if chaos_metrics:
            for ticker, m in chaos_metrics.items():
                vol_pct = m.get('volatility_pct', 0.5)
                hawkes_instance.set_ticker_volatility(ticker, vol_pct)

        results = {}
        total_bullish = 0
        total_bearish = 0
        fitted_count = 0

        for i, ticker in enumerate(all_tickers):
            try:
                candles = self._download_intraday_candles(
                    ticker, interval_minutes=interval_minutes, days_back=days_back
                )
                if not candles or len(candles.get('prices', [])) < 20:
                    results[ticker] = {'error': 'no_data', 'fitted': False}
                    continue

                prices = candles['prices']
                timestamps = candles['timestamps']

                # Если timestamps пустые или нулевые — генерируем равномерные
                if not timestamps or timestamps[0] == 0:
                    now_ts = time.time()
                    n = len(prices)
                    timestamps = [now_ts - (n - 1 - i) * (interval_minutes * 60)
                                  for i in range(n)]

                # Фидим в Hawkes
                for p, ts in zip(prices, timestamps):
                    hawkes_instance.update_price(ticker, p, ts)

                # Обучаем с current_time = последняя точка
                current_time = timestamps[-1] if timestamps[-1] > 0 else time.time()
                hawkes_instance.fit(ticker, current_time)

                # Статистика
                events = hawkes_instance._events.get(ticker, {'bullish': [], 'bearish': []})
                params = hawkes_instance._params.get(ticker, {})

                n_bull = len(events.get('bullish', []))
                n_bear = len(events.get('bearish', []))
                total_bullish += n_bull
                total_bearish += n_bear

                eta_bull = params.get('eta_bull', 0)
                eta_bear = params.get('eta_bear', 0)
                threshold = hawkes_instance._get_threshold(ticker)

                if eta_bull > 0 or eta_bear > 0:
                    fitted_count += 1

                results[ticker] = {
                    'n_candles': len(prices),
                    'events_bullish': n_bull,
                    'events_bearish': n_bear,
                    'threshold': threshold,
                    'eta_bull': eta_bull,
                    'eta_bear': eta_bear,
                    'fitted': (eta_bull > 0 or eta_bear > 0),
                }

                if (i + 1) % 10 == 0:
                    logger.info(f"  Hawkes bootstrap: {i+1}/{len(all_tickers)} тикеров обработано")

                # Rate limit между тикерами
                time.sleep(self.rate_limit)

            except Exception as e:
                logger.debug(f"Hawkes bootstrap {ticker}: {e}")
                results[ticker] = {'error': str(e), 'fitted': False}
                continue

        # Сводная статистика
        stats = hawkes_instance.get_stats()
        logger.info(
            f"✅ Hawkes bootstrap завершён: "
            f"fitted={fitted_count}/{len(all_tickers)}, "
            f"bullish_events={total_bullish}, "
            f"bearish_events={total_bearish}, "
            f"tickers_tracked={stats['tickers_tracked']}, "
            f"tickers_fitted={stats['tickers_fitted']}"
        )

        # 🆕 Детальный лог топ-5 тикеров по числу событий
        sorted_results = sorted(
            [(t, r) for t, r in results.items() if r.get('fitted')],
            key=lambda x: x[1].get('events_bullish', 0) + x[1].get('events_bearish', 0),
            reverse=True
        )[:5]
        if sorted_results:
            logger.info("📊 Топ-5 тикеров по числу событий Hawkes:")
            for ticker, r in sorted_results:
                logger.info(
                    f"  {ticker}: bull={r['events_bullish']}, bear={r['events_bearish']}, "
                    f"η_bull={r['eta_bull']:.2f}, η_bear={r['eta_bear']:.2f}, "
                    f"thr={r['threshold']:.4f}, candles={r['n_candles']}"
                )

        return results

    def _merge_history(self, existing: Dict, new: Dict) -> Dict:
        """Объединение старой и новой истории (с сохранением OHLC)."""
        merged = {}
        all_tickers = set(list(existing.keys()) + list(new.keys()))

        for ticker in all_tickers:
            old = existing.get(ticker, {})
            new_data = new.get(ticker, {})

            old_prices = old.get('prices', [])
            old_ts = old.get('timestamps', [])
            new_prices = new_data.get('prices', [])
            new_ts = new_data.get('timestamps', [])

            if new_prices:
                # Объединяем, убирая дубликаты по timestamp
                combined_ts = old_ts + new_ts
                combined_prices = old_prices + new_prices
                combined_volumes = old.get('volumes', []) + new_data.get('volumes', [])
                combined_highs = old.get('highs', []) + new_data.get('highs', [])
                combined_lows = old.get('lows', []) + new_data.get('lows', [])
                combined_opens = old.get('opens', []) + new_data.get('opens', [])

                # Убираем дубликаты
                seen = set()
                uniq_ts, uniq_prices, uniq_volumes = [], [], []
                uniq_highs, uniq_lows, uniq_opens = [], [], []
                # Если в old нет OHLC — берём close как fallback
                if not combined_highs:
                    combined_highs = combined_prices
                if not combined_lows:
                    combined_lows = combined_prices
                if not combined_opens:
                    combined_opens = combined_prices

                for i, ts in enumerate(combined_ts):
                    if ts not in seen:
                        seen.add(ts)
                        uniq_ts.append(ts)
                        uniq_prices.append(combined_prices[i])
                        uniq_volumes.append(combined_volumes[i] if i < len(combined_volumes) else 0)
                        uniq_highs.append(combined_highs[i] if i < len(combined_highs) else combined_prices[i])
                        uniq_lows.append(combined_lows[i] if i < len(combined_lows) else combined_prices[i])
                        uniq_opens.append(combined_opens[i] if i < len(combined_opens) else combined_prices[i])

                merged[ticker] = {
                    'prices': uniq_prices[-self.max_points:],
                    'highs': uniq_highs[-self.max_points:],
                    'lows': uniq_lows[-self.max_points:],
                    'opens': uniq_opens[-self.max_points:],
                    'volumes': uniq_volumes[-self.max_points:],
                    'timestamps': uniq_ts[-self.max_points:],
                }
            elif old_prices:
                merged[ticker] = old

        return merged

    def _save_history(self, history: Dict):
        """Сохранение истории в файл."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(history, f)
            logger.info(f"История сохранена: {len(history)} тикеров → {self.history_file}")

            # Также сохраняем в формате technical_core (data/price_history.json)
            # чтобы TechnicalTraderCore.load_history() мог её загрузить при старте
            tc_history = {}
            for ticker, hist in history.items():
                prices = hist.get('prices', [])
                if len(prices) < 2:
                    continue
                tc_history[ticker] = {
                    'prices': prices[-100:],
                    'volumes': hist.get('volumes', [])[-100:],
                    'timestamps': hist.get('timestamps', [])[-100:],
                    'max_length': 100,
                }
            tc_file = "data/price_history.json"
            with open(tc_file, 'w') as f:
                json.dump(tc_history, f)
            logger.info(f"История для technical_core: {len(tc_history)} тикеров → {tc_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения истории: {e}")

    def _feed_to_technical_core(self, history: Dict):
        """Передача истории в technical_core для расчёта индикаторов."""
        if not self.tech_core:
            return
        
        for ticker, hist in history.items():
            prices = hist.get('prices', [])
            volumes = hist.get('volumes', [])
            if len(prices) < 2:
                continue
            
            self.tech_core.price_history[ticker] = {
                'prices': prices[-100:],
                'volumes': volumes[-100:],
                'timestamps': [datetime.fromisoformat(t) if isinstance(t, str) else datetime.now()
                              for t in hist.get('timestamps', [])[-100:]],
                'max_length': 100
            }
            # Сбрасываем кэш индикаторов
            if ticker in self.tech_core.indicators_cache:
                del self.tech_core.indicators_cache[ticker]
        
        logger.info(f"Передано в technical_core: {len(history)} тикеров")

    def _calculate_hurst(self, prices: 'np.ndarray') -> float:
        """
        R/S анализ для расчёта показателя Херста.
        H > 0.5 — персистентный (трендовый)
        H < 0.5 — антиперсистентный (mean-reverting)
        H ≈ 0.5 — случайное блуждание
        """
        import numpy as np
        try:
            log_returns = np.diff(np.log(prices[prices > 0]))
            n_total = len(log_returns)
            if n_total < 50:
                return 0.5

            # Размеры окон: геометрическая прогрессия
            min_n = 10
            max_n = n_total // 4
            if max_n < min_n * 2:
                return 0.5

            ns = np.unique(np.logspace(
                np.log10(min_n), np.log10(max_n), num=15
            ).astype(int))
            ns = ns[ns >= min_n]

            rs_values = []
            for n in ns:
                n_blocks = n_total // n
                if n_blocks < 1:
                    continue
                rs_block = []
                for i in range(n_blocks):
                    block = log_returns[i*n:(i+1)*n]
                    mean = block.mean()
                    cum_dev = np.cumsum(block - mean)
                    R = cum_dev.max() - cum_dev.min()
                    S = block.std(ddof=1)
                    if S > 0:
                        rs_block.append(R / S)
                if rs_block:
                    rs_values.append((n, np.mean(rs_block)))

            if len(rs_values) < 4:
                return 0.5

            ns_arr = np.array([x[0] for x in rs_values])
            rs_arr = np.array([x[1] for x in rs_values])
            log_n = np.log(ns_arr)
            log_rs = np.log(rs_arr)

            # Линейная регрессия
            slope = np.polyfit(log_n, log_rs, 1)[0]
            return float(max(0.0, min(1.0, slope)))
        except Exception:
            return 0.5

    def _calculate_correlation_dimension(self, prices: 'np.ndarray') -> float:
        """
        Корреляционная размерность D₂ (Grassberger-Procaccia).
        D₂ < 3 — детерминированный хаос.
        D₂ > 5 — близко к шуму.
        """
        import numpy as np
        try:
            log_returns = np.diff(np.log(prices[prices > 0]))
            log_returns = log_returns - log_returns.mean()
            if len(log_returns) < 200:
                return 2.5  # default

            # Takens embedding: tau=4, m=3 (упрощённо)
            tau = 4
            m = 3
            n = len(log_returns) - (m - 1) * tau
            if n < 100:
                return 2.5

            embedded = np.zeros((n, m))
            for i in range(m):
                embedded[:, i] = log_returns[i*tau : i*tau + n]

            # Subsample для скорости
            if n > 2000:
                idx = np.random.choice(n, 2000, replace=False)
                embedded = embedded[idx]
                n = 2000

            # Попарные расстояния
            from scipy.spatial.distance import pdist
            dists = pdist(embedded)

            if len(dists) == 0:
                return 2.5

            # Корреляционный интеграл C(r)
            r_min = np.percentile(dists, 1)
            r_max = np.percentile(dists, 90)
            rs = np.logspace(np.log10(r_min), np.log10(r_max), num=20)
            c_values = np.array([np.sum(dists < r) / len(dists) for r in rs])

            # Наклон в log-log (линейная область)
            mask = (c_values > 0.001) & (c_values < 0.3)
            if mask.sum() < 4:
                return 2.5

            log_r = np.log(rs[mask])
            log_c = np.log(c_values[mask])
            slope = np.polyfit(log_r, log_c, 1)[0]
            return float(max(0.5, min(10.0, slope)))
        except Exception:
            return 2.5

    def _calculate_chaos_metrics(self, history: Dict):
        """
        Расчёт расширенных хаос-метрик для всех тикеров в history.
        Метрики: Херст (R/S), D₂, RQA (RR, DET, L_max, LAM),
                 volatility, skewness, kurtosis, ATR.
        Кэширует результаты в data/chaos_metrics_cache.json.
        """
        try:
            import numpy as np
            from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
            from scipy.spatial.distance import pdist
        except ImportError:
            logger.warning("scipy недоступен — расширенные хаос-метрики не рассчитаны. "
                          "Установите: pip install scipy")
            return

        # Все тикеры из history
        tickers_to_analyze = list(history.keys())
        # Обязательно добавляем индексы если есть
        for idx in self.INDEX_TICKERS:
            if idx in history and idx not in tickers_to_analyze:
                tickers_to_analyze.append(idx)

        # Тикеры из портфеля
        try:
            with open('data/portfolio_state.json', 'r') as f:
                portfolio = json.load(f)
                for t in portfolio.get('positions', {}).keys():
                    if t in history and t not in tickers_to_analyze:
                        tickers_to_analyze.append(t)
        except Exception:
            pass

        metrics = {}
        for ticker in tickers_to_analyze:
            if ticker not in history:
                continue
            prices = history[ticker].get('prices', [])
            highs = history[ticker].get('highs', [])
            lows = history[ticker].get('lows', [])
            if len(prices) < 100:
                continue

            prices_arr = np.array(prices, dtype=float)
            prices_arr = prices_arr[prices_arr > 0]
            if len(prices_arr) < 100:
                continue

            log_returns = np.diff(np.log(prices_arr))
            if len(log_returns) < 30:
                continue

            # Базовые
            vol_pct = float(np.std(log_returns) * 100)
            skew_v = float(scipy_skew(log_returns))
            kurt_v = float(scipy_kurt(log_returns, fisher=True))
            momentum = float((prices_arr[-1] / prices_arr[-min(24, len(prices_arr))] - 1) * 100)

            # ATR (14h)
            atr_pct = 0.0
            try:
                if highs and lows and len(highs) >= 14:
                    h = np.array(highs[-50:], dtype=float)
                    l = np.array(lows[-50:], dtype=float)
                    p_prev = np.array(prices[-51:-1] if len(prices) > 50 else prices[:-1], dtype=float)
                    n = min(len(h), len(l), len(p_prev))
                    tr = np.maximum(h[-n:] - l[-n:],
                                    np.maximum(np.abs(h[-n:] - p_prev[-n:]),
                                               np.abs(l[-n:] - p_prev[-n:])))
                    atr_pct = float(np.mean(tr[-14:]) / prices_arr[-1] * 100) if prices_arr[-1] > 0 else 0
            except Exception:
                atr_pct = 0.0

            # Херст
            hurst = self._calculate_hurst(prices_arr)
            # D₂
            d2 = self._calculate_correlation_dimension(prices_arr)
            # RQA
            rqa = self._calculate_rqa(log_returns)

            metrics[ticker] = {
                'n_points': int(len(prices_arr)),
                'volatility_pct': vol_pct,
                'momentum_24h': momentum,
                'hurst': hurst,
                'fractal_dim': d2,
                'skew': skew_v,
                'kurtosis': kurt_v,
                'atr_pct': atr_pct,
                'rqa_RR': rqa['RR'],
                'rqa_DET': rqa['DET'],
                'rqa_L_max': rqa['L_max'],
                'rqa_LAM': rqa['LAM'],
                'last_price': float(prices_arr[-1]),
                'last_update': datetime.now().isoformat(),
            }
            logger.info(f"  {ticker}: H={hurst:.3f}, D₂={d2:.3f}, "
                       f"vol={vol_pct:.3f}%, kurt={kurt_v:.1f}, "
                       f"DET={rqa['DET']:.2f}, L_max={rqa['L_max']}, "
                       f"ATR={atr_pct:.2f}%, n={len(prices_arr)}")

        # Сохраняем
        try:
            with open(self.chaos_cache_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Хаос-метрики сохранены: {len(metrics)} тикеров → {self.chaos_cache_file}")

            # 🆕 Калибровка Хокса: устанавливаем per-ticker thresholds
            try:
                from core.hawkes_signal import hawkes_signal
                if hawkes_signal:
                    for ticker, m in metrics.items():
                        hawkes_signal.set_ticker_volatility(ticker, m['volatility_pct'])
                    logger.info(f"Per-ticker Hawkes thresholds установлены для {len(metrics)} тикеров")
            except Exception as e:
                logger.debug(f"Не удалось откалибровать Хокс: {e}")
        except Exception as e:
            logger.warning(f"Ошибка сохранения хаос-метрик: {e}")

    def _calculate_rqa(self, log_returns: np.ndarray, tau: int = 1, m: int = 5,
                       eps_mult: float = 2.0) -> Dict:
        """Recurrence Quantification Analysis: RR, DET, L_max, LAM."""
        try:
            n_total = len(log_returns)
            if n_total < 100:
                return {'RR': 0.0, 'DET': 0.0, 'L_max': 0, 'LAM': 0.0}
            # Центрируем
            lr = log_returns - log_returns.mean()
            n = n_total - (m - 1) * tau
            if n < 100:
                return {'RR': 0.0, 'DET': 0.0, 'L_max': 0, 'LAM': 0.0}
            embedded = np.zeros((n, m))
            for i in range(m):
                embedded[:, i] = lr[i * tau:i * tau + n]
            # Subsample для скорости
            if n > 1500:
                idx = np.random.choice(n, 1500, replace=False)
                idx = np.sort(idx)
                embedded = embedded[idx]
                n = 1500
            from scipy.spatial.distance import cdist
            D = cdist(embedded, embedded)
            eps = np.std(lr) * eps_mult
            if eps <= 0:
                return {'RR': 0.0, 'DET': 0.0, 'L_max': 0, 'LAM': 0.0}
            R = (D < eps).astype(int)
            np.fill_diagonal(R, 0)
            total = R.sum()
            if total == 0:
                return {'RR': 0.0, 'DET': 0.0, 'L_max': 0, 'LAM': 0.0}
            RR = total / (n * n)
            # Диагональные линии
            diag_lens = []
            for offset in range(1, n):
                d = np.diagonal(R, offset=offset)
                cur = 0
                for v in d:
                    if v == 1:
                        cur += 1
                    else:
                        if cur >= 2:
                            diag_lens.append(cur)
                        cur = 0
                if cur >= 2:
                    diag_lens.append(cur)
            # Вертикальные линии
            vert_lens = []
            for j in range(n):
                col = R[:, j]
                cur = 0
                for v in col:
                    if v == 1:
                        cur += 1
                    else:
                        if cur >= 2:
                            vert_lens.append(cur)
                        cur = 0
                if cur >= 2:
                    vert_lens.append(cur)
            det_pts = sum(l for l in diag_lens)
            lam_pts = sum(l for l in vert_lens)
            DET = det_pts / total if total > 0 else 0
            LAM = lam_pts / total if total > 0 else 0
            L_max = max(diag_lens) if diag_lens else 0
            return {'RR': float(RR), 'DET': float(DET), 'L_max': int(L_max), 'LAM': float(LAM)}
        except Exception as e:
            logger.debug(f"RQA error: {e}")
            return {'RR': 0.0, 'DET': 0.0, 'L_max': 0, 'LAM': 0.0}

    def get_chaos_metrics(self) -> Dict:
        """Загрузка кэшированных хаос-метрик."""
        try:
            with open(self.chaos_cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _is_chaos_cache_stale(self) -> bool:
        """Проверка устаревания кэша хаос-метрик (старше 24 часов)."""
        try:
            if not os.path.exists(self.chaos_cache_file):
                return True
            mtime = os.path.getmtime(self.chaos_cache_file)
            age_hours = (datetime.now().timestamp() - mtime) / 3600
            return age_hours > 24
        except Exception:
            return True


# Синглтон
history_loader = HistoryLoader()
