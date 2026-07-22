#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆕 v16 Фаза 3.1: ML-стратегия прогноза цены как 5-я фаза Entry Cascading.

Логика:
  • Использует PricePredictionCascade для прогноза цены на N часов вперёд
  • Если predicted_return > threshold → усиливает confidence сигнала
  • Если predicted_return < -threshold → блокирует вход (даже если Hawkes/tech дали BUY)
  • Возвращает adjustment: multiplier для confidence и флаг block

Параметры из settings.json → entry_cascading.ml_strategy:
  - enabled: true
  - horizon_hours: 1          (прогноз на 1 час)
  - min_predicted_return_pct: 0.5   (минимум +0.5% для усиления)
  - max_predicted_return_pct: 5.0   (максимум для нормализации confidence)
  - block_if_predicted_loss_pct: 1.0 (блокировать если прогноз < -1%)

Обратная совместимость: если price_predictor недоступен — стратегия отключена.
"""
from typing import Dict, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("ML_STRATEGY")


class MLPriceStrategy:
    """ML-стратегия прогноза цены для 5-й фазы Entry Cascading."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.horizon_hours = self.config.get('horizon_hours', 1)
        self.min_predicted_return_pct = self.config.get('min_predicted_return_pct', 0.5)
        self.max_predicted_return_pct = self.config.get('max_predicted_return_pct', 5.0)
        self.block_if_predicted_loss_pct = self.config.get('block_if_predicted_loss_pct', 1.0)

        # Статистика
        self.stats = {
            'evaluations': 0,
            'boosts': 0,
            'blocks': 0,
            'avg_predicted_return': 0.0,
        }

        logger.info(
            f"MLPriceStrategy init: horizon={self.horizon_hours}h, "
            f"boost>{self.min_predicted_return_pct}%, "
            f"block<{-self.block_if_predicted_loss_pct}%"
        )

    def evaluate(self, ticker: str, current_price: float,
                 price_predictor, chaos_metrics: Dict = None,
                 hawkes_forecast: Dict = None) -> Dict:
        """
        Оценка тикера через ML-прогноз.

        Returns:
            {
                'action': 'BOOST' | 'BLOCK' | 'NEUTRAL',
                'confidence_multiplier': float,  # 0.0 - 1.5
                'predicted_return_pct': float,
                'predicted_price': float,
                'reason': str,
            }
        """
        if not self.enabled or not price_predictor:
            return {
                'action': 'NEUTRAL',
                'confidence_multiplier': 1.0,
                'predicted_return_pct': 0.0,
                'predicted_price': current_price,
                'reason': 'ml_strategy disabled or no predictor',
            }

        self.stats['evaluations'] += 1

        try:
            # 🆕 Извлекаем hurst из chaos_metrics для передачи в price_predictor
            hurst = 0.5
            if chaos_metrics:
                hurst = chaos_metrics.get('hurst', 0.5)

            prediction = price_predictor.predict(
                ticker=ticker,
                entry_price=current_price,
                current_price=current_price,
                hold_time_hours=self.horizon_hours,
                hawkes_signal=(hawkes_forecast or {}).get('net_signal', 0),
                hurst=hurst,
            )

            predicted_price = prediction.get('predicted_price', current_price)
            if predicted_price <= 0 or current_price <= 0:
                return {
                    'action': 'NEUTRAL',
                    'confidence_multiplier': 1.0,
                    'predicted_return_pct': 0.0,
                    'predicted_price': current_price,
                    'reason': 'invalid prediction',
                }

            predicted_return_pct = ((predicted_price / current_price) - 1) * 100

            # Обновляем статистику
            n = self.stats['evaluations']
            self.stats['avg_predicted_return'] = (
                (self.stats['avg_predicted_return'] * (n - 1) + predicted_return_pct) / n
            )

            # БЛОКИРОВКА: прогноз сильного убытка
            if predicted_return_pct < -self.block_if_predicted_loss_pct:
                self.stats['blocks'] += 1
                return {
                    'action': 'BLOCK',
                    'confidence_multiplier': 0.0,
                    'predicted_return_pct': predicted_return_pct,
                    'predicted_price': predicted_price,
                    'reason': f'ML block: predicted {predicted_return_pct:+.2f}% < '
                              f'-{self.block_if_predicted_loss_pct}%',
                }

            # УСИЛЕНИЕ: прогноз прибыли
            if predicted_return_pct > self.min_predicted_return_pct:
                self.stats['boosts'] += 1
                # Множитель confidence от 1.0 до 1.5
                boost_ratio = min(
                    predicted_return_pct / self.max_predicted_return_pct,
                    1.0
                )
                multiplier = 1.0 + 0.5 * boost_ratio
                return {
                    'action': 'BOOST',
                    'confidence_multiplier': multiplier,
                    'predicted_return_pct': predicted_return_pct,
                    'predicted_price': predicted_price,
                    'reason': f'ML boost: predicted +{predicted_return_pct:.2f}% '
                              f'(multiplier {multiplier:.2f})',
                }

            # НЕЙТРАЛЬНО
            return {
                'action': 'NEUTRAL',
                'confidence_multiplier': 1.0,
                'predicted_return_pct': predicted_return_pct,
                'predicted_price': predicted_price,
                'reason': f'ML neutral: predicted {predicted_return_pct:+.2f}%',
            }

        except Exception as e:
            logger.debug(f"ML strategy error for {ticker}: {e}")
            return {
                'action': 'NEUTRAL',
                'confidence_multiplier': 1.0,
                'predicted_return_pct': 0.0,
                'predicted_price': current_price,
                'reason': f'ml error: {e}',
            }

    def get_stats(self) -> Dict:
        return self.stats.copy()


# Синглтон
try:
    import json
    with open('config/settings.json', 'r', encoding='utf-8') as f:
        _settings = json.load(f)
    _ml_config = _settings.get('entry_cascading', {}).get('ml_strategy', {})
    ml_price_strategy = MLPriceStrategy(_ml_config)
except Exception as e:
    ml_price_strategy = None
    logger.warning(f"MLPriceStrategy не инициализирован: {e}")