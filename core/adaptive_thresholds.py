#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆕 v16 Фаза 3.3: Adaptive Thresholds — динамическая подстройка порогов.

Логика:
  • Анализирует win_rate за последние N сделок
  • Если win_rate < min_threshold → ужесточаем min_confidence
  • Если win_rate > max_threshold → ослабляем min_confidence
  • Период пересчёта — каждые N циклов

Параметры из settings.json → adaptive_thresholds:
  - enabled: true
  - window_size: 20           (сколько последних сделок анализировать)
  - min_win_rate: 0.40        (ниже — ужесточаем)
  - max_win_rate: 0.60        (выше — ослабляем)
  - adjustment_step: 0.05     (шаг изменения min_confidence)
  - min_confidence_floor: 0.30  (минимум min_confidence)
  - min_confidence_ceiling: 0.70 (максимум min_confidence)

Обратная совместимость: если выключено — пороги не меняются.
"""
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger("ADAPTIVE")


class AdaptiveThresholds:
    """Динамическая подстройка порогов на основе win_rate."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.window_size = self.config.get('window_size', 20)
        self.min_win_rate = self.config.get('min_win_rate', 0.40)
        self.max_win_rate = self.config.get('max_win_rate', 0.60)
        self.adjustment_step = self.config.get('adjustment_step', 0.05)
        self.min_confidence_floor = self.config.get('min_confidence_floor', 0.30)
        self.min_confidence_ceiling = self.config.get('min_confidence_ceiling', 0.70)

        # Текущее значение min_confidence
        self.current_min_confidence = self.config.get('initial_min_confidence', 0.50)

        # История сделок для расчёта win_rate
        self.trade_results: List[bool] = []  # True = win, False = loss

        logger.info(
            f"AdaptiveThresholds init: window={self.window_size}, "
            f"win_rate=[{self.min_win_rate}, {self.max_win_rate}], "
            f"min_conf={self.current_min_confidence:.2f}"
        )

    def record_trade(self, pnl: float):
        """Запись результата сделки для win_rate."""
        if not self.enabled:
            return
        self.trade_results.append(pnl > 0)
        # Ограничиваем историю
        if len(self.trade_results) > self.window_size * 2:
            self.trade_results = self.trade_results[-self.window_size:]

    def recalculate(self) -> float:
        """
        Пересчёт min_confidence на основе win_rate.
        Возвращает новое значение min_confidence.
        """
        if not self.enabled or len(self.trade_results) < self.window_size:
            return self.current_min_confidence

        recent = self.trade_results[-self.window_size:]
        wins = sum(1 for r in recent if r)
        win_rate = wins / len(recent)

        old_conf = self.current_min_confidence

        if win_rate < self.min_win_rate:
            # Слишком много убытков — ужесточаем
            self.current_min_confidence = min(
                self.min_confidence_ceiling,
                self.current_min_confidence + self.adjustment_step
            )
            if abs(self.current_min_confidence - old_conf) > 1e-6:
                logger.warning(
                    f"📉 [ADAPTIVE] win_rate={win_rate:.0%} < {self.min_win_rate:.0%} → "
                    f"ужесточаем min_conf {old_conf:.2f} → {self.current_min_confidence:.2f}"
                )
        elif win_rate > self.max_win_rate:
            # Хороший win_rate — можно ослабить
            self.current_min_confidence = max(
                self.min_confidence_floor,
                self.current_min_confidence - self.adjustment_step
            )
            if abs(self.current_min_confidence - old_conf) > 1e-6:
                logger.info(
                    f"📈 [ADAPTIVE] win_rate={win_rate:.0%} > {self.max_win_rate:.0%} → "
                    f"ослабляем min_conf {old_conf:.2f} → {self.current_min_confidence:.2f}"
                )

        return self.current_min_confidence

    def get_stats(self) -> Dict:
        if not self.trade_results:
            return {'win_rate': 0, 'trades_recorded': 0, 'current_min_confidence': self.current_min_confidence}
        wins = sum(1 for r in self.trade_results if r)
        return {
            'win_rate': wins / len(self.trade_results),
            'trades_recorded': len(self.trade_results),
            'current_min_confidence': self.current_min_confidence,
        }


# Синглтон
try:
    import json
    with open('config/settings.json', 'r', encoding='utf-8') as f:
        _settings = json.load(f)
    _adaptive_config = _settings.get('adaptive_thresholds', {})
    adaptive_thresholds = AdaptiveThresholds(_adaptive_config)
except Exception as e:
    adaptive_thresholds = None
    logger.warning(f"AdaptiveThresholds не инициализирован: {e}")
