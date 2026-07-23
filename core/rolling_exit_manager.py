#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вариант D: Rolling Exit Manager.

Почасовая переоценка удерживаемых позиций. Каждый цикл для каждой позиции:
  1. Расчёт sell_score (6 компонент)
  2. Динамический порог по hold_time
  3. Жёсткие стопы (stop-loss, profit-taking)
  4. Фазовый выход (50%/30%/20%)

Все настройки в config/settings.json → rolling_exit.
"""
import time
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger("ROLLING_EXIT")


@dataclass
class PositionState:
    ticker: str
    entry_price: float
    entry_time: float
    entry_qty: float
    hurst: float
    atr_pct: float
    kurtosis: float
    rqa_det: float
    rqa_l_max: int
    hawkes_baseline_signal: float
    ms_baseline_imbalance: float
    dynamic_target: float
    stop_loss_pct: float
    stop_loss_price: float
    phase: int = 1  # 1 = full position, 2 = 50% sold, 3 = 80% sold, 0 = closed
    last_evaluation_cycle: int = 0
    last_sell_score: float = 0.0
    last_evaluation_reason: str = ""
    phase2_pending_cycles: int = 0  # ждёт подтверждения для phase 2
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0


@dataclass
class ExitDecision:
    ticker: str
    action: str  # 'HOLD' | 'SELL_FULL' | 'SELL_PARTIAL_50' | 'SELL_PARTIAL_30' | 'SELL_FINAL'
    sell_score: float
    threshold: float
    reason: str
    qty_to_sell: float = 0.0
    confidence: float = 0.0


class RollingExitManager:
    """Rolling exit manager (Вариант D) — почасовая переоценка позиций."""

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        self.enabled = config.get('enabled', True)
        self.eval_interval = config.get('evaluation_interval_cycles', 1)

        self.weights = config.get('sell_score_components', {})
        self.thresholds_cfg = config.get('thresholds_by_hold_time', {})
        self.hard_stops_cfg = config.get('hard_stops', {})
        self.phase_cfg = config.get('phase_exit', {})
        self.min_hold_hours = config.get('minimum_hold_hours_before_sell', 2)
        self.min_pnl_for_profit_sell = config.get('min_pnl_pct_for_profit_sell', 0.5)

        # 🆕 v16.3: Конфигурируемые threshold adjustments (вместо хардкода)
        ta_cfg = config.get('threshold_adjustment', {})
        self.hurst_persistent_thr = ta_cfg.get('hurst_persistent_threshold', 0.55)
        self.hurst_antipersistent_thr = ta_cfg.get('hurst_antipersistent_threshold', 0.45)
        self.hurst_adj = ta_cfg.get('hurst_adjustment', 0.05)
        self.low_det_thr = ta_cfg.get('low_det_threshold', 0.20)
        self.low_det_adj = ta_cfg.get('low_det_adjustment', 0.05)
        self.threshold_min = ta_cfg.get('threshold_min', 0.20)
        self.threshold_max = ta_cfg.get('threshold_max', 0.85)

        self._positions: Dict[str, PositionState] = {}

        logger.info(f"RollingExitManager init: enabled={self.enabled}, "
                    f"eval_interval={self.eval_interval}")

    def on_buy(self, ticker: str, price: float, qty: float,
               chaos_metrics: Dict, hawkes_signal_val: float,
               ms_imbalance: float, strategy_stop_loss_pct: float = None):
        """Регистрация новой позиции."""
        atr_pct = chaos_metrics.get('atr_pct', 1.0)
        kurt = chaos_metrics.get('kurtosis', 10)

        # Динамический stop-loss с учётом тяжёлых хвостов
        base_stop = self.hard_stops_cfg.get('base_stop_loss_pct', 2.5)
        kurt_factor = self.hard_stops_cfg.get('kurtosis_penalty_factor', 0.02)
        kurt_baseline = self.hard_stops_cfg.get('kurtosis_baseline', 3)
        kurt_max = self.hard_stops_cfg.get('kurtosis_penalty_max', 5.0)
        kurt_penalty = min(kurt_max, max(0, (kurt - kurt_baseline) * kurt_factor))
        stop_loss_pct = strategy_stop_loss_pct or (base_stop + kurt_penalty)
        stop_loss_price = price * (1 - stop_loss_pct / 100)

        # Динамическая цель
        dynamic_target = price * (1 + 2 * atr_pct / 100)

        pos = PositionState(
            ticker=ticker,
            entry_price=price,
            entry_time=time.time(),
            entry_qty=qty,
            hurst=chaos_metrics.get('hurst', 0.5),
            atr_pct=atr_pct,
            kurtosis=kurt,
            rqa_det=chaos_metrics.get('rqa_DET', 0.3),
            rqa_l_max=chaos_metrics.get('rqa_L_max', 6),
            hawkes_baseline_signal=hawkes_signal_val,
            ms_baseline_imbalance=ms_imbalance,
            dynamic_target=dynamic_target,
            stop_loss_pct=stop_loss_pct,
            stop_loss_price=stop_loss_price,
        )
        self._positions[ticker] = pos
        logger.info(f"[ROLL_EXIT] {ticker} registered: entry={price:.2f}, "
                    f"stop={stop_loss_price:.2f} (-{stop_loss_pct:.2f}%), "
                    f"target={dynamic_target:.2f}, "
                    f"kurt={kurt:.1f}, DET={pos.rqa_det:.2f}, L_max={pos.rqa_l_max}")

    def evaluate(self, ticker: str, current_price: float, current_cycle: int,
                 get_pred_1h, get_hawkes_signal, get_indicators, get_microstructure,
                 ) -> Optional[ExitDecision]:
        """
        Оценка позиции для решения HOLD/SELL.

        Аргументы — callback-функции:
          get_pred_1h(ticker) -> {p_down, p_up, p_sideways, action}
          get_hawkes_signal(ticker) -> float (net signal)
          get_indicators(ticker) -> {rsi, rsi_short, momentum_1h, momentum_4h, bb_position,
                                     local_max_6h, distance_from_local_max_pct, atr_pct, ...}
          get_microstructure(ticker) -> {imbalance, spread_pct, ...}
        """
        if not self.enabled:
            return None
        pos = self._positions.get(ticker)
        if not pos:
            return None

        # Проверка интервала
        if current_cycle - pos.last_evaluation_cycle < self.eval_interval:
            return None
        pos.last_evaluation_cycle = current_cycle

        now = time.time()
        hold_hours = (now - pos.entry_time) / 3600
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100

        # Минимум hold-время (кроме hard stop)
        if hold_hours < self.min_hold_hours:
            # Только hard stop может сработать раньше
            hard_decision = self._check_hard_stops(pos, current_price, pnl_pct, hold_hours)
            if hard_decision:
                return hard_decision
            return ExitDecision(
                ticker=ticker, action='HOLD', sell_score=0, threshold=0,
                reason=f"min_hold {hold_hours:.1f}h < {self.min_hold_hours}h"
            )

        # Сбор сигналов
        pred_1h = get_pred_1h(ticker) or {}
        hawkes_now = get_hawkes_signal(ticker) or 0
        ind = get_indicators(ticker) or {}
        ms = get_microstructure(ticker) or {}

        # ─── COMPUTE SELL SCORE ───
        sell_score, components = self._compute_sell_score(
            pos, current_price, pnl_pct, pred_1h, hawkes_now, ind, ms
        )

        # ─── HARD STOPS ───
        hard_decision = self._check_hard_stops(pos, current_price, pnl_pct, hold_hours)
        if hard_decision:
            hard_decision.sell_score = sell_score
            hard_decision.reason = (hard_decision.reason + " | " +
                                    " | ".join(components) if components else hard_decision.reason)
            return hard_decision

        # ─── DYNAMIC THRESHOLD ───
        threshold = self._dynamic_threshold(pos, hold_hours, current_price, pnl_pct)

        # ─── DECISION ───
        decision = self._decide(pos, sell_score, threshold, current_price, pnl_pct,
                                hold_hours, components)
        return decision

    def _compute_sell_score(self, pos: PositionState, current_price: float,
                            pnl_pct: float, pred_1h: Dict, hawkes_now: float,
                            ind: Dict, ms: Dict) -> Tuple[float, List[str]]:
        """Расчёт composite sell score и компонент."""
        components = []
        score = 0.0

        # (1) Прогноз разворота на 1ч — самый важный сигнал
        w_pred = self.weights.get('weight_pred_1h_down', 0.30)
        p_down_1h = pred_1h.get('p_down', 0.33)
        score += w_pred * p_down_1h
        if p_down_1h > 0.50:
            components.append(f"P(down_1h)={p_down_1h:.2f}")

        # (2) Цена у локального максимума
        w_lmax = self.weights.get('weight_at_local_max', 0.20)
        local_max_6h = ind.get('local_max_6h', current_price)
        dist_from_max = ind.get('distance_from_local_max_pct', -100)
        if local_max_6h > 0 and dist_from_max > -0.3:
            score += w_lmax
            components.append(f"at_local_max_6h ({dist_from_max:+.2f}%)")

        # (3) Hawkes сменил знак
        w_hawkes = self.weights.get('weight_hawkes_reversal', 0.20)
        if pos.hawkes_baseline_signal > 0 and hawkes_now < 0:
            score += w_hawkes
            components.append(f"hawkes_reversal ({pos.hawkes_baseline_signal:+.2f}→{hawkes_now:+.2f})")
        elif pos.hawkes_baseline_signal < 0 and hawkes_now > 0 and pnl_pct < 0:
            # Bearish→bullish reversal при убытке — наоборот, держать (отскок)
            score -= w_hawkes * 0.5
            components.append(f"hawkes_recovery ({pos.hawkes_baseline_signal:+.2f}→{hawkes_now:+.2f})")

        # (4) Микроструктура развернулась
        w_ms = self.weights.get('weight_ms_reversal', 0.15)
        imb_now = ms.get('imbalance', 0)
        if pos.ms_baseline_imbalance > 0.3 and imb_now < -0.3:
            score += w_ms
            components.append(f"ms_reversal ({pos.ms_baseline_imbalance:+.2f}→{imb_now:+.2f})")

        # (5) RSI перекупленность + momentum замедляется
        w_rsi = self.weights.get('weight_rsi_momentum_decay', 0.15)
        rsi_short = ind.get('rsi_short', 50)
        mom_1h = ind.get('momentum_1h', 0)
        mom_4h = ind.get('momentum_4h', 0)
        if rsi_short > 70 and mom_4h != 0 and mom_1h < mom_4h / 2:
            score += w_rsi
            components.append(f"rsi_overbought_momentum_decay (RSI={rsi_short:.0f})")

        # (6) Штраф за сильный тренд вверх (удержание)
        penalty = self.weights.get('penalty_strong_up_trend', 0.25)
        p_up_1h = pred_1h.get('p_up', 0.33)
        if p_up_1h > 0.55 and mom_1h > 0 and mom_4h > 0:
            score -= penalty
            components.append(f"strong_up_trend P(up_1h)={p_up_1h:.2f}")

        score = max(0, min(1, score))
        return score, components

    def _check_hard_stops(self, pos: PositionState, current_price: float,
                          pnl_pct: float, hold_hours: float) -> Optional[ExitDecision]:
        """Жёсткие стопы: stop-loss, profit-taking, max hold time."""
        # Stop-loss
        if pnl_pct <= -pos.stop_loss_pct:
            return ExitDecision(
                ticker=pos.ticker, action='SELL_FULL',
                sell_score=1.0, threshold=0,
                qty_to_sell=pos.entry_qty,
                reason=f"hard_stop_loss PnL={pnl_pct:+.2f}% ≤ -{pos.stop_loss_pct:.2f}%"
            )
        # Price упал ниже stop_loss_price
        if current_price <= pos.stop_loss_price:
            return ExitDecision(
                ticker=pos.ticker, action='SELL_FULL',
                sell_score=1.0, threshold=0,
                qty_to_sell=pos.entry_qty,
                reason=f"stop_loss_price {current_price:.2f} ≤ {pos.stop_loss_price:.2f}"
            )

        # Profit-taking: при PnL > threshold + sell_score > min
        pt_threshold = self.hard_stops_cfg.get('profit_taking_threshold_pct', 5.0)
        pt_min_score = self.hard_stops_cfg.get('profit_taking_min_sell_score', 0.40)
        if pnl_pct >= pt_threshold:
            # Только если есть хотя бы слабый сигнал разворота
            # (если sell_score ещё не посчитан — продаём при pt_threshold + 50% ATR)
            atr_extension = pos.atr_pct * 0.5
            if pnl_pct >= pt_threshold + atr_extension:
                return ExitDecision(
                    ticker=pos.ticker, action='SELL_FULL',
                    sell_score=0.5, threshold=pt_min_score,
                    qty_to_sell=pos.entry_qty,
                    reason=f"profit_taking_extended PnL={pnl_pct:+.2f}%"
                )

        # Max hold time hard cap
        max_hold = self.thresholds_cfg.get('max_hold_hours_hard_cap', 120)
        if hold_hours >= max_hold:
            return ExitDecision(
                ticker=pos.ticker, action='SELL_FULL',
                sell_score=1.0, threshold=0,
                qty_to_sell=pos.entry_qty,
                reason=f"max_hold_cap {hold_hours:.1f}h ≥ {max_hold}h"
            )
        return None

    def _dynamic_threshold(self, pos: PositionState, hold_hours: float,
                            current_price: float, pnl_pct: float) -> float:
        """Динамический порог продажи по hold_time и Hurst."""
        early_h = self.thresholds_cfg.get('early_hours', 4)
        early_t = self.thresholds_cfg.get('early_threshold', 0.65)
        mid_h = self.thresholds_cfg.get('mid_hours', 24)
        mid_t = self.thresholds_cfg.get('mid_threshold', 0.55)
        late_h = self.thresholds_cfg.get('late_hours', 72)
        late_t = self.thresholds_cfg.get('late_threshold', 0.45)
        force_t = self.thresholds_cfg.get('force_threshold', 0.30)

        if hold_hours < early_h:
            t = early_t
        elif hold_hours < mid_h:
            t = mid_t
        elif hold_hours < late_h:
            t = late_t
        else:
            t = force_t

        # Hurst-корректировка: персистентные тикеры — выше порог (тренд скорее продолжится)
        if pos.hurst > self.hurst_persistent_thr:
            t += self.hurst_adj
        elif pos.hurst < self.hurst_antipersistent_thr:
            t -= self.hurst_adj

        # Низкий DET — выше порог (шумные сигналы)
        if pos.rqa_det < self.low_det_thr:
            t += self.low_det_adj

        return max(self.threshold_min, min(self.threshold_max, t))

    def _decide(self, pos: PositionState, sell_score: float, threshold: float,
                current_price: float, pnl_pct: float, hold_hours: float,
                components: List[str]) -> ExitDecision:
        """Принятие решения с учётом фазового выхода."""
        pos.last_sell_score = sell_score
        reason_str = " | ".join(components) if components else "no_signal"
        pos.last_evaluation_reason = reason_str

        phase_enabled = self.phase_cfg.get('enabled', True)
        p1_ratio = self.phase_cfg.get('phase1_ratio', 0.5)
        p2_ratio = self.phase_cfg.get('phase2_ratio', 0.3)
        p3_ratio = self.phase_cfg.get('phase3_ratio', 0.2)
        p2_confirm = self.phase_cfg.get('phase2_confirm_cycles', 1)
        p3_atr_mult = self.phase_cfg.get('phase3_trailing_atr_mult', 1.5)

        # ─── ФАЗОВЫЙ ВЫХОД ───
        if phase_enabled:
            # Phase 3: trailing stop активен
            if pos.phase == 3:
                # Обновляем trailing stop
                new_trailing = max(pos.trailing_stop_price,
                                   current_price * (1 - p3_atr_mult * pos.atr_pct / 100))
                pos.trailing_stop_price = new_trailing
                if current_price <= pos.trailing_stop_price:
                    qty = pos.entry_qty * p3_ratio
                    return ExitDecision(
                        ticker=pos.ticker, action='SELL_FINAL',
                        sell_score=sell_score, threshold=threshold,
                        qty_to_sell=qty,
                        reason=f"phase3_trailing_stop {current_price:.2f} ≤ {pos.trailing_stop_price:.2f}"
                    )
                # Обновляем dynamic target
                if current_price > pos.dynamic_target:
                    pos.dynamic_target = current_price * (1 + 1.5 * pos.atr_pct / 100)
                return ExitDecision(
                    ticker=pos.ticker, action='HOLD',
                    sell_score=sell_score, threshold=threshold,
                    reason=f"phase3_hold trailing={pos.trailing_stop_price:.2f}"
                )

            # Phase 2: ждём подтверждения
            if pos.phase == 2:
                if sell_score >= threshold:
                    pos.phase2_pending_cycles += 1
                    if pos.phase2_pending_cycles >= p2_confirm:
                        qty = pos.entry_qty * p2_ratio
                        pos.phase = 3
                        pos.trailing_stop_price = current_price * (1 - p3_atr_mult * pos.atr_pct / 100)
                        return ExitDecision(
                            ticker=pos.ticker, action='SELL_PARTIAL_30',
                            sell_score=sell_score, threshold=threshold,
                            qty_to_sell=qty,
                            reason=f"phase2_confirmed score={sell_score:.2f}≥{threshold:.2f}"
                        )
                else:
                    pos.phase2_pending_cycles = 0  # сброс если сигнал ослаб
                return ExitDecision(
                    ticker=pos.ticker, action='HOLD',
                    sell_score=sell_score, threshold=threshold,
                    reason=f"phase2_hold pending={pos.phase2_pending_cycles}/{p2_confirm}"
                )

        # ─── ОБЫЧНОЕ РЕШЕНИЕ (Phase 1) ───

        # 🆕 v16 Фаза 3.2: Trailing stop для Phase 1 — фиксация прибыли при развороте
        # Если цена выросла > activation_pct от входа, активируем trailing
        trailing_cfg = self.phase_cfg.get('trailing_phase1', {})
        trailing_p1_enabled = trailing_cfg.get('enabled', True)
        trailing_p1_activation = trailing_cfg.get('activation_pct', 2.0)
        trailing_p1_distance = trailing_cfg.get('distance_pct', 1.0)

        if trailing_p1_enabled and pos.entry_price > 0:
            profit_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            if profit_pct >= trailing_p1_activation:
                # Обновляем trailing stop
                new_trailing = max(pos.trailing_stop_price,
                                   current_price * (1 - trailing_p1_distance / 100))
                pos.trailing_stop_price = new_trailing
                pos.trailing_stop_active = True

                if current_price <= pos.trailing_stop_price and pos.trailing_stop_price > pos.entry_price:
                    # Trailing сработал — продаём всю позицию (profit-taking)
                    return ExitDecision(
                        ticker=pos.ticker, action='SELL_FULL',
                        sell_score=sell_score, threshold=threshold,
                        qty_to_sell=pos.entry_qty,
                        reason=f"phase1_trailing_stop {current_price:.2f} ≤ {pos.trailing_stop_price:.2f} "
                               f"(profit was +{profit_pct:.1f}%)"
                    )

        if sell_score >= threshold:
            if phase_enabled:
                # Phase 1 → продаём 50%, переходим в phase 2
                qty = pos.entry_qty * p1_ratio
                pos.phase = 2
                pos.phase2_pending_cycles = 0
                return ExitDecision(
                    ticker=pos.ticker, action='SELL_PARTIAL_50',
                    sell_score=sell_score, threshold=threshold,
                    qty_to_sell=qty,
                    reason=f"phase1_sell score={sell_score:.2f}≥{threshold:.2f} | {reason_str}"
                )
            else:
                return ExitDecision(
                    ticker=pos.ticker, action='SELL_FULL',
                    sell_score=sell_score, threshold=threshold,
                    qty_to_sell=pos.entry_qty,
                    reason=f"sell score={sell_score:.2f}≥{threshold:.2f} | {reason_str}"
                )

        # HOLD — обновляем dynamic_target
        if current_price > pos.dynamic_target:
            pos.dynamic_target = current_price * (1 + 1.5 * pos.atr_pct / 100)
            logger.debug(f"[ROLL_EXIT] {pos.ticker} dynamic_target → {pos.dynamic_target:.2f}")

        return ExitDecision(
            ticker=pos.ticker, action='HOLD',
            sell_score=sell_score, threshold=threshold,
            reason=f"hold score={sell_score:.2f}<{threshold:.2f} | {reason_str}"
        )

    def on_position_closed(self, ticker: str):
        """Удаление позиции при полном закрытии."""
        if ticker in self._positions:
            del self._positions[ticker]
            logger.debug(f"[ROLL_EXIT] {ticker} position removed from tracking")

    def on_partial_fill(self, ticker: str, qty_sold: float):
        """Обновление qty при частичной продаже."""
        if ticker in self._positions:
            self._positions[ticker].entry_qty -= qty_sold
            if self._positions[ticker].entry_qty <= 0:
                self.on_position_closed(ticker)

    def get_active_positions(self) -> Dict[str, dict]:
        """Возвращает слепки активных позиций для дашборда."""
        return {
            t: {
                'entry_price': p.entry_price,
                'entry_qty': p.entry_qty,
                'hold_hours': (time.time() - p.entry_time) / 3600,
                'phase': p.phase,
                'stop_loss_price': p.stop_loss_price,
                'dynamic_target': p.dynamic_target,
                'last_sell_score': p.last_sell_score,
                'last_reason': p.last_evaluation_reason,
                'hurst': p.hurst,
                'rqa_det': p.rqa_det,
                'rqa_l_max': p.rqa_l_max,
                'kurtosis': p.kurtosis,
                'trailing_stop': p.trailing_stop_price,
            }
            for t, p in self._positions.items()
        }

    def get_stats(self) -> Dict:
        return {
            'enabled': self.enabled,
            'active_positions': len(self._positions),
            'phase_distribution': dict(defaultdict(int, {
                p.phase: 1 for p in self._positions.values()
            })),
        }


# ─── СИНГЛТОН ───
_config = {}
try:
    _settings_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'settings.json'
    )
    if os.path.exists(_settings_path):
        with open(_settings_path, 'r', encoding='utf-8') as _f:
            _settings = json.load(_f)
            _config = _settings.get('rolling_exit', {})
except Exception as e:
    logger.warning(f"Не удалось загрузить rolling_exit config: {e}")

rolling_exit_manager = RollingExitManager(_config)
