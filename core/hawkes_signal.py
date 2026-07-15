#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Процесс Хокса для прогнозирования кластеризации событий.

События: |log-return| > threshold (по умолчанию 1%).
Обучение: EM-алгоритм, rolling window.
Прогноз: E[bullish events] и E[bearish events] на горизонт h часов.
"""
import time
import numpy as np
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional
import json
import os

from utils.logger import get_logger

logger = get_logger("HAWKES")


class HawkesSignalGenerator:
    """Процесс Хокса с экспоненциальным ядром для торговых сигналов.

    Поддерживает per-ticker event_threshold_pct (калибровка по волатильности).
    """

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        self.default_threshold = config.get('event_threshold_pct', 0.01)
        self.window_size = config.get('window_size', 4000)
        self.refit_interval = config.get('refit_interval', 500)
        self.forecast_horizon = config.get('forecast_horizon_hours', 48)
        self.max_iter = config.get('max_iter', 50)

        # Per-ticker thresholds (из конфига или рассчитанные по волатильности)
        pt_cfg = config.get('per_ticker_thresholds', {})
        self._pt_split = pt_cfg.get('vol_threshold_split', 0.5)  # в %
        self._pt_low_mult = pt_cfg.get('low_vol_multiplier', 0.3)
        self._pt_high_mult = pt_cfg.get('high_vol_multiplier', 0.5)
        self._pt_overrides = {}  # ручные overrides {ticker: threshold}
        self._ticker_volatility = {}  # кэш {ticker: vol_pct}
        self._ticker_thresholds = {}  # итоговые thresholds {ticker: threshold}

        # История цен по тикерам
        self._price_history = defaultdict(lambda: deque(maxlen=self.window_size))
        self._events = defaultdict(lambda: {'bullish': [], 'bearish': []})
        self._params = {}
        self._last_fit_cycle = {}

        logger.info(f"HawkesSignalGenerator: default_threshold={self.default_threshold}, "
                    f"window={self.window_size}, refit={self.refit_interval}, "
                    f"per_ticker=ON (split={self._pt_split}%)")

    def set_ticker_volatility(self, ticker: str, vol_pct: float):
        """Установка волатильности тикера для автоматической калибровки порога."""
        self._ticker_volatility[ticker] = vol_pct
        # Авто-расчёт порога: k × vol, где k зависит от уровня волатильности
        if vol_pct < self._pt_split:
            thr = self._pt_low_mult * vol_pct / 100
        else:
            thr = self._pt_high_mult * vol_pct / 100
        thr = max(thr, 0.0005)  # минимум 0.05%
        # Ручной override имеет приоритет
        if ticker in self._pt_overrides:
            thr = self._pt_overrides[ticker]
        self._ticker_thresholds[ticker] = thr

    def set_ticker_threshold_override(self, ticker: str, threshold: float):
        """Ручной override порога для конкретного тикера."""
        self._pt_overrides[ticker] = threshold
        self._ticker_thresholds[ticker] = threshold

    def _get_threshold(self, ticker: str) -> float:
        """Получение порога событий для тикера."""
        return self._ticker_thresholds.get(ticker, self.default_threshold)

    def get_branching_ratio(self, ticker: str, direction: str = 'bull') -> float:
        """Branching ratio η для оценки силы кластеризации."""
        if ticker not in self._params:
            return 0.0
        return self._params[ticker].get(f'eta_{direction}', 0.0)

    def update_price(self, ticker: str, price: float, timestamp: float = None):
        """Обновление цены и обнаружение событий (per-ticker threshold)."""
        if timestamp is None:
            timestamp = time.time()
        if price <= 0:
            return

        history = self._price_history[ticker]
        history.append((timestamp, price))

        threshold = self._get_threshold(ticker)

        # Проверяем событие (нужно минимум 2 точки)
        if len(history) >= 2:
            prev_ts, prev_price = history[-2]
            curr_ts, curr_price = history[-1]
            dt = curr_ts - prev_ts
            if dt > 0 and prev_price > 0:
                log_ret = np.log(curr_price / prev_price)
                if log_ret > threshold:
                    self._events[ticker]['bullish'].append(curr_ts)
                elif log_ret < -threshold:
                    self._events[ticker]['bearish'].append(curr_ts)

        # Ограничиваем историю событий
        for direction in ['bullish', 'bearish']:
            evts = self._events[ticker][direction]
            if len(evts) > self.window_size:
                self._events[ticker][direction] = evts[-self.window_size:]

    def fit(self, ticker: str, current_time: float):
        """EM-алгоритм для оценки параметров процесса Хокса.
        Время конвертируется из секунд в часы для стабильности EM.
        """
        for direction in ['bull', 'bear']:
            events_key = 'bullish' if direction == 'bull' else 'bearish'
            events = np.array(sorted(self._events[ticker][events_key]), dtype=float)
            n = len(events)
            if n < 10:
                self._params.setdefault(ticker, {})[f'mu_{direction}'] = 0.001
                self._params.setdefault(ticker, {})[f'eta_{direction}'] = 0.0
                self._params.setdefault(ticker, {})[f'beta_{direction}'] = 1.0
                continue

            # 🆕 v14.2: Конвертируем время из секунд в часы
            # Раньше: dt = 20000с → beta = 1/20000 = 0.00005 → exp(-0.00005*20000) = 0.37
            # Теперь: dt = 5.5ч → beta = 1/5.5 = 0.18 → exp(-0.18*5.5) = 0.37 (то же самое)
            # Но EM-алгоритм работает стабильнее в часах
            events_hours = events / 3600.0
            current_time_hours = current_time / 3600.0

            T = current_time_hours - events_hours[0] if len(events_hours) > 0 else 1.0
            T = max(T, 1.0)

            # Инициализация
            mu = n / T * 0.5
            eta = 0.5
            beta = 1.0 / max(np.mean(np.diff(events_hours)), 1e-6) if n > 1 else 1.0

            # EM итерации (в часах)
            for iteration in range(self.max_iter):
                # E-step: A[i] = sum_{j<i} exp(-beta*(t_i - t_j))
                A = np.zeros(n)
                for i in range(1, n):
                    dt = events_hours[i] - events_hours[i-1]
                    A[i] = np.exp(-beta * dt) * (1 + A[i-1])

                lam = mu + eta * beta * A
                lam = np.maximum(lam, 1e-10)
                p_bg = mu / lam
                p_trig = 1.0 - p_bg

                # M-step
                mu_new = np.sum(p_bg) / T
                eta_new = np.sum(p_trig) / n
                eta_new = min(max(eta_new, 0.01), 0.95)

                # B[i] = sum (t_i - t_j) * exp(-beta*(t_i-t_j))
                B = np.zeros(n)
                for i in range(1, n):
                    dt = events_hours[i] - events_hours[i-1]
                    B[i] = dt * np.exp(-beta * dt) * (1 + A[i-1]) + np.exp(-beta * dt) * B[i-1]

                denom = np.sum(p_trig * B / np.maximum(A, 1e-10))
                if denom > 0:
                    beta_new = np.sum(p_trig) / denom
                    beta_new = max(beta_new, 1e-4)
                else:
                    beta_new = beta

                # Сходимость
                if abs(mu_new - mu) < 1e-6 and abs(eta_new - eta) < 1e-6:
                    break
                mu, eta, beta = mu_new, eta_new, beta_new

            # 🆕 v14.2: Прямое присваивание (setdefault не перезаписывал существующие значения)
            if ticker not in self._params:
                self._params[ticker] = {}
            self._params[ticker][f'mu_{direction}'] = float(mu)
            self._params[ticker][f'eta_{direction}'] = float(eta)
            self._params[ticker][f'beta_{direction}'] = float(beta)

    def forecast(self, ticker: str, current_time: float,
                 horizon: float = None) -> Dict:
        """Прогноз ожидаемого числа событий на горизонт h (часы)."""
        if horizon is None:
            horizon = self.forecast_horizon

        if ticker not in self._params:
            return {'bull_expected': 0.0, 'bear_expected': 0.0,
                    'net_signal': 0.0, 'prob_bull': 0.0, 'prob_bear': 0.0}

        params = self._params[ticker]
        result = {}

        # 🆕 v14.2: Конвертируем в часы (beta теперь в часах)
        current_time_h = current_time / 3600.0
        horizon_h = horizon  # уже в часах

        for direction in ['bull', 'bear']:
            mu = params.get(f'mu_{direction}', 0.001)
            eta = params.get(f'eta_{direction}', 0.0)
            beta = params.get(f'beta_{direction}', 1.0)

            # Вклад от прошлых событий (в часах)
            events = np.array(self._events[ticker].get(
                'bullish' if direction == 'bull' else 'bearish', []), dtype=float)
            past = events[events <= current_time]
            if len(past) > 0:
                past_h = past / 3600.0
                triggered = eta * np.sum(
                    (1 - np.exp(-beta * horizon_h)) *
                    np.exp(-beta * (current_time_h - past_h))
                )
            else:
                triggered = 0.0

            background = mu * horizon_h
            expected = background + triggered
            prob = 1.0 - np.exp(-expected) if expected > 0 else 0.0

            result[f'{direction}_expected'] = float(expected)
            result[f'prob_{direction}'] = float(prob)

        result['net_signal'] = result.get('bull_expected', 0) - result.get('bear_expected', 0)
        return result

    def get_signal(self, ticker: str, current_time: float = None) -> float:
        """Возвращает net signal (bull - bear) для каскадного предсказателя."""
        if current_time is None:
            current_time = time.time()
        fc = self.forecast(ticker, current_time)
        return fc.get('net_signal', 0.0)

    def should_refit(self, ticker: str, cycle_count: int) -> bool:
        """Проверка необходимости переобучения."""
        last = self._last_fit_cycle.get(ticker, -self.refit_interval)
        if cycle_count - last >= self.refit_interval:
            self._last_fit_cycle[ticker] = cycle_count
            return True
        return False

    def get_params(self, ticker: str) -> Dict:
        return self._params.get(ticker, {})

    def get_stats(self) -> Dict:
        return {
            'tickers_tracked': len(self._price_history),
            'tickers_fitted': len(self._params),
            'total_bullish_events': sum(len(e['bullish']) for e in self._events.values()),
            'total_bearish_events': sum(len(e['bearish']) for e in self._events.values()),
        }


# Синглтон с загрузкой конфига
_config = {}
try:
    with open('config/settings.json', 'r', encoding='utf-8') as f:
        _settings = json.load(f)
        _config = _settings.get('hawkes', {})
except Exception:
    pass

hawkes_signal = HawkesSignalGenerator(_config)
