#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆕 v16.12: Post-sale tracking — слежение за тикером после продажи.

Логика:
  1. При продаже тикера регистрируется post-sale tracking
  2. Каждый цикл проверяется цена проданного тикера
  3. Через tracking_hours (по умолчанию 1 час) вычисляется результат:
     - Цена упала → продажа была правильной → +reward модели
     - Цена выросла → продали рано → -penalty модели
  4. Результат записывается как experience в RL-модель

Параметры из settings.json → post_sale_tracking:
  - enabled: true
  - tracking_hours: 1.0     (горизонт слежения)
  - reward_correct_sell: 0.5  (продали вовремя — цена упала)
  - penalty_premature_sell: -0.5  (продали рано — цена выросла)
  - min_price_change_pct: 0.1  (минимальное изменение цены для reward/penalty)
"""
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger("POST_SALE")


class PostSaleTracker:
    """Слежение за тикерами после продажи для оценки качества выхода."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.tracking_hours = self.config.get('tracking_hours', 1.0)
        self.reward_correct = self.config.get('reward_correct_sell', 0.5)
        self.penalty_premature = self.config.get('penalty_premature_sell', -0.5)
        self.min_change_pct = self.config.get('min_price_change_pct', 0.1)

        # Активные tracking записи: {ticker: {...}}
        self._tracking = {}

        # Статистика
        self.stats = {
            'total_tracked': 0,
            'correct_sells': 0,
            'premature_sells': 0,
            'neutral_sells': 0,
            'total_reward': 0.0,
        }

        logger.info(f"PostSaleTracker init: tracking_hours={self.tracking_hours}, "
                    f"reward={self.reward_correct}, penalty={self.penalty_premature}")

    def register_sale(self, ticker: str, sell_price: float, sell_time: float = None):
        """Регистрация продажи для последующего tracking."""
        if not self.enabled:
            return

        if sell_time is None:
            sell_time = time.time()

        self._tracking[ticker] = {
            'sell_price': sell_price,
            'sell_time': sell_time,
            'tracking_hours': self.tracking_hours,
            'evaluated': False,
        }

        self.stats['total_tracked'] += 1
        logger.debug(f"[POST_SALE] {ticker} registered: sell_price={sell_price:.2f}, "
                    f"tracking for {self.tracking_hours}h")

    def check_and_evaluate(self, ticker: str, current_price: float,
                           current_time: float = None) -> Optional[Dict]:
        """
        Проверка и оценка tracking для тикера.

        Returns:
            None — если tracking ещё не завершён
            {'ticker': ..., 'result': 'correct'|'premature'|'neutral',
             'reward': float, 'price_change_pct': float} — если оценён
        """
        if not self.enabled or ticker not in self._tracking:
            return None

        record = self._tracking[ticker]
        if record.get('evaluated', False):
            return None

        if current_time is None:
            current_time = time.time()

        elapsed_hours = (current_time - record['sell_time']) / 3600
        if elapsed_hours < self.tracking_hours:
            return None  # ещё рано

        # Время истекло — оцениваем
        sell_price = record['sell_price']
        if sell_price <= 0 or current_price <= 0:
            self._cleanup(ticker)
            return None

        price_change_pct = ((current_price - sell_price) / sell_price) * 100

        result = 'neutral'
        reward = 0.0

        if price_change_pct < -self.min_change_pct:
            # Цена упала → продажа была правильной
            result = 'correct'
            reward = self.reward_correct
            self.stats['correct_sells'] += 1
        elif price_change_pct > self.min_change_pct:
            # Цена выросла → продали рано
            result = 'premature'
            reward = self.penalty_premature
            self.stats['premature_sells'] += 1
        else:
            # Нейтрально — цена не изменилась
            self.stats['neutral_sells'] += 1

        self.stats['total_reward'] += reward
        record['evaluated'] = True

        logger.info(
            f"[POST_SALE] {ticker} evaluated: result={result}, "
            f"sell_price={sell_price:.2f}, current={current_price:.2f}, "
            f"change={price_change_pct:+.2f}%, reward={reward:+.2f}"
        )

        result_dict = {
            'ticker': ticker,
            'result': result,
            'reward': reward,
            'price_change_pct': price_change_pct,
            'sell_price': sell_price,
            'current_price': current_price,
            'elapsed_hours': elapsed_hours,
        }

        self._cleanup(ticker)
        return result_dict

    def _cleanup(self, ticker: str):
        """Удаление завершённого tracking."""
        if ticker in self._tracking:
            del self._tracking[ticker]

    def get_active_tracking(self) -> Dict:
        """Возвращает активные tracking записи."""
        return dict(self._tracking)

    def get_stats(self) -> Dict:
        return self.stats.copy()


# Синглтон
try:
    import json
    with open('config/settings.json', 'r', encoding='utf-8') as f:
        _settings = json.load(f)
    _pst_config = _settings.get('post_sale_tracking', {})
    post_sale_tracker = PostSaleTracker(_pst_config)
except Exception as e:
    post_sale_tracker = None
    logger.warning(f"PostSaleTracker не инициализирован: {e}")
