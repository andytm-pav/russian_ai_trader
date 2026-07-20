#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆕 v16 Фаза 4.1: Multi-Timeframe Analysis — подтверждение сигналов на нескольких ТФ.

Логика:
  • Анализирует RSI/momentum на 3 таймфреймах: 5м, 1ч, 1д
  • Если все 3 ТФ согласованы (RSI < 50 = перепроданность на всех) → усиливаем сигнал
  • Если старший ТФ противоречит младшим → ослабляем confidence
  • Возвращает multiplier от 0.5 до 1.3

Параметры из settings.json → multi_timeframe:
  - enabled: true
  - timeframes: [5, 60, 1440]  # минуты
  - agreement_boost: 1.3        # множитель при полном согласии
  - conflict_penalty: 0.7       # множитель при конфликте

Обратная совместимость: если выключено — multiplier = 1.0.
"""
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger("MULTI_TF")


class MultiTimeframeAnalyzer:
    """Мульти-таймфрейм анализ для подтверждения сигналов."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.timeframes = self.config.get('timeframes', [5, 60, 1440])  # 5м, 1ч, 1д
        self.agreement_boost = self.config.get('agreement_boost', 1.3)
        self.conflict_penalty = self.config.get('conflict_penalty', 0.7)

        logger.info(
            f"MultiTimeframeAnalyzer init: timeframes={self.timeframes}, "
            f"boost={self.agreement_boost}, penalty={self.conflict_penalty}"
        )

    def analyze(self, ticker: str, indicators_by_tf: Dict[int, Dict]) -> Dict:
        """
        Анализ индикаторов на нескольких ТФ.

        Args:
            indicators_by_tf: {5: {rsi: 25, momentum: -2}, 60: {rsi: 35, ...}, 1440: {rsi: 40, ...}}

        Returns:
            {
                'multiplier': float,
                'agreement': 'FULL' | 'PARTIAL' | 'CONFLICT',
                'signals_by_tf': {5: 'OVERSOLD', 60: 'OVERSOLD', 1440: 'NEUTRAL'},
                'reason': str,
            }
        """
        if not self.enabled or not indicators_by_tf:
            return {'multiplier': 1.0, 'agreement': 'DISABLED', 'reason': 'disabled'}

        signals = {}
        for tf in self.timeframes:
            ind = indicators_by_tf.get(tf)
            if not ind:
                signals[tf] = 'NO_DATA'
                continue

            rsi = ind.get('rsi', 50)
            mom = ind.get('momentum', 0)

            if rsi < 30 and mom < 0:
                signals[tf] = 'OVERSOLD'
            elif rsi > 70 and mom > 0:
                signals[tf] = 'OVERBOUGHT'
            elif 45 <= rsi <= 55:
                signals[tf] = 'NEUTRAL'
            elif rsi < 50 and mom < 0:
                signals[tf] = 'BEARISH'
            elif rsi > 50 and mom > 0:
                signals[tf] = 'BULLISH'
            else:
                signals[tf] = 'NEUTRAL'

        # Анализ согласия
        valid_signals = [s for s in signals.values() if s not in ('NO_DATA', 'NEUTRAL')]
        if not valid_signals:
            return {
                'multiplier': 1.0,
                'agreement': 'NO_SIGNAL',
                'signals_by_tf': signals,
                'reason': 'no clear signals on any TF',
            }

        # Проверяем согласие: все в одну сторону (bullish/oversold) или (bearish/overbought)
        bullish_signals = sum(1 for s in valid_signals if s in ('OVERSOLD', 'BULLISH'))
        bearish_signals = sum(1 for s in valid_signals if s in ('OVERBOUGHT', 'BEARISH'))

        if bullish_signals == len(valid_signals):
            # Полное согласие на rise/oversold
            return {
                'multiplier': self.agreement_boost,
                'agreement': 'FULL_BULLISH',
                'signals_by_tf': signals,
                'reason': f'full agreement: {bullish_signals}/{len(valid_signals)} bullish',
            }

        if bearish_signals == len(valid_signals):
            return {
                'multiplier': self.agreement_boost,
                'agreement': 'FULL_BEARISH',
                'signals_by_tf': signals,
                'reason': f'full agreement: {bearish_signals}/{len(valid_signals)} bearish',
            }

        # Конфликт: есть и bullish, и bearish
        if bullish_signals > 0 and bearish_signals > 0:
            return {
                'multiplier': self.conflict_penalty,
                'agreement': 'CONFLICT',
                'signals_by_tf': signals,
                'reason': f'conflict: {bullish_signals} bullish vs {bearish_signals} bearish',
            }

        # Частичное согласие
        return {
            'multiplier': 1.0,
            'agreement': 'PARTIAL',
            'signals_by_tf': signals,
            'reason': f'partial: {bullish_signals} bull, {bearish_signals} bear, '
                     f'{len(valid_signals) - bullish_signals - bearish_signals} neutral',
        }


# Синглтон
try:
    import json
    with open('config/settings.json', 'r', encoding='utf-8') as f:
        _settings = json.load(f)
    _mtf_config = _settings.get('multi_timeframe', {})
    multi_timeframe_analyzer = MultiTimeframeAnalyzer(_mtf_config)
except Exception as e:
    multi_timeframe_analyzer = None
    logger.warning(f"MultiTimeframeAnalyzer не инициализирован: {e}")
