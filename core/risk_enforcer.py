#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆕 v16 Фаза 1.4: Risk Enforcer — принудительные стоп-лоссы и запрет усреднения убытка.

Логика:
  1. Перед каждым BUY проверяем: если по этому тикеру уже есть позиция в убытке > N% → ЗАПРЕТ.
  2. Каждый цикл проверяем все позиции: если убыток > M% → принудительный SELL всей позиции.
  3. Логируем все события для аудита.

Параметры из settings.json → risk_management:
  - max_position_loss_pct: 5.0     (стоп-лосс: продажа при убытке > 5%)
  - no_averaging_below_loss_pct: 3.0  (запрет докупки при убытке > 3%)

Обратная совместимость: если секция risk_management отсутствует, модуль отключён.
"""
import time
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("RISK_ENFORCER")


class RiskEnforcer:
    """Принудительное управление рисками позиций."""

    def __init__(self, config: Dict):
        self.config = config or {}
        self.max_position_loss_pct = self.config.get('max_position_loss_pct', 5.0)
        self.no_averaging_below_loss_pct = self.config.get('no_averaging_below_loss_pct', 3.0)
        self.enforce_stop_loss = self.config.get('enforce_stop_loss', True)
        self.enforce_no_averaging = self.config.get('enforce_no_averaging', True)

        # Статистика
        self.stats = {
            'stop_losses_triggered': 0,
            'averaging_blocked': 0,
            'total_loss_avoided_rub': 0.0,
        }

        logger.info(
            f"RiskEnforcer init: stop_loss={self.max_position_loss_pct}% "
            f"(enabled={self.enforce_stop_loss}), "
            f"no_averaging>{self.no_averaging_below_loss_pct}% "
            f"(enabled={self.enforce_no_averaging})"
        )

    def check_buy_allowed(self, ticker: str, positions: Dict, current_price: float) -> Tuple[bool, str]:
        """
        Проверка: можно ли покупать тикер (не нарушает ли правило запрета усреднения убытка).

        Returns:
            (allowed, reason)
        """
        if not self.enforce_no_averaging:
            return True, "no_averaging disabled"

        if ticker not in positions:
            return True, "new position"

        pos = positions[ticker]
        avg_price = pos.get('avg_price', 0)
        if avg_price <= 0 or current_price <= 0:
            return True, "no avg_price"

        pnl_pct = (current_price - avg_price) / avg_price * 100
        if pnl_pct < -self.no_averaging_below_loss_pct:
            reason = (f"averaging blocked: {ticker} в убытке {pnl_pct:+.2f}% "
                     f"< -{self.no_averaging_below_loss_pct}% (avg={avg_price:.2f}, "
                     f"now={current_price:.2f})")
            self.stats['averaging_blocked'] += 1
            logger.warning(f"🚫 [RISK] {reason}")
            return False, reason

        return True, f"ok (pnl={pnl_pct:+.2f}%)"

    def get_stop_loss_sells(self, positions: Dict, prices: Dict) -> List[Dict]:
        """
        Возвращает список позиций для принудительной продажи по стоп-лоссу.

        Returns:
            [{'ticker': ..., 'qty': ..., 'price': ..., 'loss_pct': ..., 'reason': ...}, ...]
        """
        if not self.enforce_stop_loss:
            return []

        sells = []
        for ticker, pos in positions.items():
            avg_price = pos.get('avg_price', 0)
            current_price = prices.get(ticker, 0)
            qty = pos.get('qty', 0)

            if avg_price <= 0 or current_price <= 0 or qty <= 0:
                continue

            pnl_pct = (current_price - avg_price) / avg_price * 100
            if pnl_pct < -self.max_position_loss_pct:
                sells.append({
                    'ticker': ticker,
                    'qty': qty,  # продаём всю позицию
                    'price': current_price,
                    'avg_price': avg_price,
                    'loss_pct': pnl_pct,
                    'reason': f'stop_loss_{pnl_pct:.1f}%',
                })
                self.stats['stop_losses_triggered'] += 1
                loss_amount = (current_price - avg_price) * qty
                # loss_amount отрицательный → "избежать" дальнейших потерь (оценка)
                self.stats['total_loss_avoided_rub'] += abs(loss_amount) * 0.5  # эвристика
                logger.warning(
                    f"🚨 [RISK] STOP-LOSS {ticker}: pnl={pnl_pct:+.2f}% "
                    f"< -{self.max_position_loss_pct}% → SELL {qty} @ {current_price:.2f} "
                    f"(avg={avg_price:.2f})"
                )

        return sells

    def get_stats(self) -> Dict:
        return self.stats.copy()
