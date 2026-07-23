#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Каскадный предсказатель цены.

После покупки тикера модель прогнозирует цену на уменьшающихся горизонтах:
  День 0-5 (цена в плюсе):  прогноз 5 дней вперёд
  Цена в минусе:            прогноз 2 дня → 1 день → 1 час → 15 мин → 1 мин
  Продажа перед снижением.

Логика:
  1. Свежая позиция (0-5 дней), PnL ≥ 0:
     → Прогноз 5 дней (горизонт Ляпунова)
     → Если P(снижение) > 50% → готовимся продать
     
  2. Позиция в минусе, hold < 2 дней:
     → Прогноз 2 дня
     → Если P(снижение) > 55% → продаём
     
  3. Позиция в минусе, hold < 1 дня:
     → Прогноз 1 день
     → Если P(снижение) > 60% → продаём
     
  4. Позиция в минусе, hold > 2 дней:
     → Прогноз 1 час
     → Если P(снижение) > 65% → продаём
     
  5. Дальше: 15 мин → 1 мин
     → Если P(снижение) > 70% → продаём немедленно

Источники прогноза по горизонтам:
  5 дней: RL-модель predictor + Хокс-сигнал + Херст (персистентность)
  2 дня: RL-модель predictor + momentum + RSI
  1 день: momentum + RSI + BB position
  1 час: микроструктура imbalance + short momentum
  15 мин: микроструктура imbalance + spread changes
  1 мин: tick momentum + spread + order flow
"""
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger("PRICE_PREDICTOR")


class PricePredictionCascade:
    """Каскадный предсказатель цены с адаптивным горизонтом."""

    def __init__(self, config: Dict = None):
        # Загрузка конфигурации каскада
        if config is None:
            config = {}
        cascade_cfg = config.get('cascade', {})

        # Максимум удержания — 5 дней (Ляпунов), НЕ 10 дней
        self.max_hold_hours = config.get('max_hold_hours', 120)

        # Каскад: (horizon_hours, name, sell_threshold, max_hold_for_level)
        # Все значения из конфига
        self.CASCADE = [
            (cascade_cfg.get('level_0_horizon_hours', 120),
             cascade_cfg.get('level_0_name', '5_days'),
             cascade_cfg.get('level_0_sell_threshold', 0.50),
             cascade_cfg.get('level_0_max_hold_hours', 120)),
            (cascade_cfg.get('level_1_horizon_hours', 48),
             cascade_cfg.get('level_1_name', '2_days'),
             cascade_cfg.get('level_1_sell_threshold', 0.55),
             cascade_cfg.get('level_1_max_hold_hours', 48)),
            (cascade_cfg.get('level_2_horizon_hours', 24),
             cascade_cfg.get('level_2_name', '1_day'),
             cascade_cfg.get('level_2_sell_threshold', 0.60),
             cascade_cfg.get('level_2_max_hold_hours', 120)),
            (cascade_cfg.get('level_3_horizon_hours', 1),
             cascade_cfg.get('level_3_name', '1_hour'),
             cascade_cfg.get('level_3_sell_threshold', 0.65),
             cascade_cfg.get('level_3_max_hold_hours', 120)),
            (cascade_cfg.get('level_4_horizon_hours', 0.25),
             cascade_cfg.get('level_4_name', '15_min'),
             cascade_cfg.get('level_4_sell_threshold', 0.70),
             cascade_cfg.get('level_4_max_hold_hours', 120)),
            (cascade_cfg.get('level_5_horizon_hours', 0.0167),
             cascade_cfg.get('level_5_name', '1_min'),
             cascade_cfg.get('level_5_sell_threshold', 0.75),
             cascade_cfg.get('level_5_max_hold_hours', 120)),
        ]

        self._predictions_history = defaultdict(list)
        self._cascade_level = {}

        # 🆕 v16.2: Параметры адаптации под D₂ (fractal_dim)
        d2_cfg = config.get('d2_adaptation', {})
        self.d2_default = d2_cfg.get('default', 2.5)
        self.d2_high_threshold = d2_cfg.get('high_threshold', 2.5)
        self.d2_low_threshold = d2_cfg.get('low_threshold', 2.0)
        self.d2_high_factor = d2_cfg.get('high_factor', 0.7)
        self.d2_low_factor = d2_cfg.get('low_factor', 1.3)

        # 🆕 v16.3: Конфигурируемые веса прогноза (вместо хардкода)
        pw = config.get('prediction_weights', {})
        self.pw_rl_defaults = pw.get('rl_defaults', {'down': 0.33, 'sideways': 0.34, 'up': 0.33})
        self.pw_tech = pw.get('tech', {})
        self.pw_ms = pw.get('microstructure', {})
        self.pw_hawkes = pw.get('hawkes', {})
        self.pw_hurst = pw.get('hurst', {})
        self.pw_levels = pw.get('level_weights', {})

        logger.info(f"PricePredictionCascade инициализирован (max_hold={self.max_hold_hours}ч = {self.max_hold_hours/24:.0f} дней)")

    def _determine_horizon(self, ticker: str, entry_price: float,
                            current_price: float, hold_time_hours: float) -> Tuple[int, float, str, float]:
        """
        Определяет, на каком горизонте прогнозировать.
        Максимум удержания — max_hold_hours (5 дней = 120ч, по Ляпунову).
        """
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        current_level = self._cascade_level.get(ticker, 0)

        # Логика каскада — все пороги из CASCADE config:
        # Уровень 0: PnL ≥ 0 → прогноз 5 дней
        # Уровень 1: PnL < 0, hold < 48ч → прогноз 2 дня
        # Уровень 2: PnL < 0, hold < 120ч (5 дней) → прогноз 1 день
        # Уровень 3: PnL < 0, hold ≥ 120ч → прогноз 1 час
        # Уровень 4: PnL < 0, hold ≥ 120ч → прогноз 15 мин
        # Уровень 5: PnL < 0, hold ≥ 120ч → прогноз 1 мин (экстренный выход)

        if pnl_pct >= 0:
            level = 0
        elif hold_time_hours < self.CASCADE[1][3]:  # level_1_max_hold_hours = 48
            level = 1
        elif hold_time_hours < self.max_hold_hours:  # < 120ч (5 дней)
            level = 2
        elif hold_time_hours < self.max_hold_hours * 1.4:  # < 168ч (7 дней)
            level = 3
        elif hold_time_hours < self.max_hold_hours * 1.7:  # < 204ч (~8.5 дней)
            level = 4
        else:
            level = 5

        # Каскад только углубляется
        level = max(level, current_level)
        self._cascade_level[ticker] = level

        horizon_hours, name, threshold, _ = self.CASCADE[level]
        return level, horizon_hours, name, threshold

    def predict(self, ticker: str, entry_price: float, current_price: float,
                hold_time_hours: float, model=None, state=None,
                indicators: Dict = None, microstructure: Dict = None,
                hawkes_signal: float = 0.0, hurst: float = 0.5,
                fractal_dim: Optional[float] = None) -> Dict:
        """
        Прогноз движения цены на адаптивном горизонте.
        
        Возвращает:
            {
                'action': 'SELL' | 'HOLD',
                'confidence': 0-1,
                'horizon_hours': float,
                'horizon_name': str,
                'cascade_level': int,
                'p_down': float,  # вероятность снижения
                'p_up': float,    # вероятность роста
                'p_sideways': float,
                'reason': str,
            }
        """
        level, horizon_hours, horizon_name, sell_threshold = self._determine_horizon(
            ticker, entry_price, current_price, hold_time_hours
        )

        # ─── 🆕 v16.2: АДАПТАЦИЯ ГОРИЗОНТА ПОД D₂ ───
        if fractal_dim is None:
            fractal_dim = self.d2_default

        d2_factor = 1.0
        if fractal_dim > self.d2_high_threshold:
            d2_factor = self.d2_high_factor
        elif fractal_dim < self.d2_low_threshold:
            d2_factor = self.d2_low_factor

        adjusted_horizon = horizon_hours * d2_factor
        adjusted_horizon = max(0.0167, adjusted_horizon)  # минимум 1 минута

        if d2_factor != 1.0:
            logger.debug(f"[D₂] {ticker}: D₂={fractal_dim:.2f} → horizon {horizon_hours:.2f}h → {adjusted_horizon:.2f}h")

        horizon_hours = adjusted_horizon

        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

        # --- ИСТОЧНИКИ ПРОГНОЗА ---

        # 1. RL-модель predictor (3 класса: down, sideways, up)
        p_down_rl = self.pw_rl_defaults.get('down', 0.33)
        p_up_rl = self.pw_rl_defaults.get('up', 0.33)
        p_sideways_rl = self.pw_rl_defaults.get('sideways', 0.34)
        if model is not None and state is not None:
            try:
                import torch
                with torch.no_grad():
                    _, _, price_pred = model.policy_net(state.unsqueeze(0))
                    probs = model.get_price_pred_probs(price_pred)
                    p_down_rl = float(probs[0])
                    p_sideways_rl = float(probs[1])
                    p_up_rl = float(probs[2])
            except Exception as e:
                logger.debug(f"RL predictor error for {ticker}: {e}")

        # 2. Технические индикаторы
        tech_down = 0.0
        tech_up = 0.0
        if indicators:
            rsi = indicators.get('rsi', 50)
            bb_pos = indicators.get('bb_position', 0.5)
            momentum = indicators.get('momentum', 0)
            
            tech_contrib = self.pw_tech.get('tech_contribution', 0.3)
            if rsi > self.pw_tech.get('rsi_overbought', 70):
                tech_down += tech_contrib
            elif rsi < self.pw_tech.get('rsi_oversold', 30):
                tech_up += tech_contrib
            
            bb_contrib = self.pw_tech.get('bb_contribution', 0.2)
            if bb_pos > self.pw_tech.get('bb_upper', 0.8):
                tech_down += bb_contrib
            elif bb_pos < self.pw_tech.get('bb_lower', 0.2):
                tech_up += bb_contrib
            
            mom_contrib = self.pw_tech.get('momentum_contribution', 0.2)
            if momentum < self.pw_tech.get('momentum_down', -2):
                tech_down += mom_contrib
            elif momentum > self.pw_tech.get('momentum_up', 2):
                tech_up += mom_contrib

        # 3. Микроструктура (для коротких горизонтов — важнее)
        ms_down = 0.0
        ms_up = 0.0
        if microstructure:
            imbalance = microstructure.get('imbalance', 0)
            spread_pct = microstructure.get('spread_pct', 0)
            
            ms_contrib = self.pw_ms.get('imbalance_contribution', 0.3)
            imb_thr = self.pw_ms.get('imbalance_threshold', 0.2)
            if imbalance > imb_thr:
                ms_up += ms_contrib * min(abs(imbalance), 1)
            elif imbalance < -imb_thr:
                ms_down += ms_contrib * min(abs(imbalance), 1)
            
            spread_thr = self.pw_ms.get('spread_threshold', 0.3)
            spread_contrib = self.pw_ms.get('spread_contribution', 0.1)
            if spread_pct > spread_thr:
                ms_down += spread_contrib

        # 4. Хокс-сигнал
        hawkes_down = 0.0
        hawkes_up = 0.0
        hawkes_contrib = self.pw_hawkes.get('contribution', 0.2)
        hawkes_bull_thr = self.pw_hawkes.get('bull_threshold', 0.5)
        hawkes_bear_thr = self.pw_hawkes.get('bear_threshold', -0.5)
        if hawkes_signal > hawkes_bull_thr:
            hawkes_up += hawkes_contrib
        elif hawkes_signal < hawkes_bear_thr:
            hawkes_down += hawkes_contrib

        # 5. Херст (персистентность)
        hurst_down = 0.0
        hurst_up = 0.0
        hurst_contrib = self.pw_hurst.get('contribution', 0.15)
        hurst_pers_thr = self.pw_hurst.get('persistent_threshold', 0.55)
        hurst_antipers_thr = self.pw_hurst.get('antipersistent_threshold', 0.45)
        if hurst > hurst_pers_thr:
            if pnl_pct >= 0:
                hurst_up += hurst_contrib
            else:
                hurst_down += hurst_contrib
        elif hurst < hurst_antipers_thr:
            if pnl_pct < 0:
                hurst_up += hurst_contrib
            else:
                hurst_down += hurst_contrib

        # --- ВЕСА ИСТОЧНИКОВ ПО ГОРИЗОНТУ ---

        if level <= 1:
            weights = self.pw_levels.get('long', {'rl': 0.40, 'tech': 0.20, 'ms': 0.10, 'hawkes': 0.15, 'hurst': 0.15})
        elif level <= 2:
            weights = self.pw_levels.get('medium', {'rl': 0.30, 'tech': 0.30, 'ms': 0.15, 'hawkes': 0.10, 'hurst': 0.15})
        elif level <= 3:
            weights = self.pw_levels.get('short', {'rl': 0.20, 'tech': 0.35, 'ms': 0.30, 'hawkes': 0.05, 'hurst': 0.10})
        else:
            weights = self.pw_levels.get('ultra_short', {'rl': 0.10, 'tech': 0.20, 'ms': 0.55, 'hawkes': 0.05, 'hurst': 0.10})

        # --- ИТОГОВЫЙ ПРОГНОЗ ---

        p_down = (p_down_rl * weights['rl'] +
                  tech_down * weights['tech'] +
                  ms_down * weights['ms'] +
                  hawkes_down * weights['hawkes'] +
                  hurst_down * weights['hurst'])

        p_up = (p_up_rl * weights['rl'] +
                tech_up * weights['tech'] +
                ms_up * weights['ms'] +
                hawkes_up * weights['hawkes'] +
                hurst_up * weights['hurst'])

        p_sideways = max(0, 1 - p_down - p_up)

        # Нормализация
        total = p_down + p_up + p_sideways
        if total > 0:
            p_down /= total
            p_up /= total
            p_sideways /= total

        # --- РЕШЕНИЕ ---

        should_sell = p_down > sell_threshold
        action = 'SELL' if should_sell else 'HOLD'
        confidence = p_down if should_sell else p_up

        reason = (f"cascade_{horizon_name} | "
                  f"PnL={pnl_pct:+.1f}% hold={hold_time_hours:.1f}ч | "
                  f"P(down)={p_down:.2f} P(up)={p_up:.2f} | "
                  f"threshold={sell_threshold}")

        if should_sell:
            logger.info(f"🎯 ПРОДАЖА {ticker}: {reason}")

        result = {
            'action': action,
            'confidence': confidence,
            'horizon_hours': horizon_hours,
            'horizon_name': horizon_name,
            'cascade_level': level,
            'p_down': p_down,
            'p_up': p_up,
            'p_sideways': p_sideways,
            'sell_threshold': sell_threshold,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }

        # Сохраняем историю прогнозов
        self._predictions_history[ticker].append(result)
        if len(self._predictions_history[ticker]) > 100:
            self._predictions_history[ticker].pop(0)

        return result

    def reset_ticker(self, ticker: str):
        """Сброс каскада при закрытии позиции."""
        if ticker in self._cascade_level:
            del self._cascade_level[ticker]
        logger.debug(f"Каскад сброшен для {ticker}")

    def get_stats(self) -> Dict:
        """Статистика каскада."""
        return {
            'active_tickers': len(self._cascade_level),
            'cascade_levels': dict(self._cascade_level),
            'total_predictions': sum(len(h) for h in self._predictions_history.values()),
        }


# Синглтон — инициализируется с конфигом при первом импорте
import json as _json
import os as _os
_pp_config = {}
try:
    _settings_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), 'config', 'settings.json')
    if _os.path.exists(_settings_path):
        with open(_settings_path, 'r') as _f:
            _settings = _json.load(_f)
            _pp_config = _settings.get('price_predictor', {})
except Exception:
    pass
price_predictor = PricePredictionCascade(_pp_config)
