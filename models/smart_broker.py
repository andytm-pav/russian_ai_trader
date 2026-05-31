"""
Главный модуль Smart Broker с интеграцией всех компонентов
"""

import json
import torch
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from core.core_technical_trader import TechnicalTraderCore
from core.risk_manager import RiskManager
from core.trading_hours_scheduler import TradingScheduler
from fetchers.moex_fetcher import MoexFetcher
from fetchers.news_fetcher import OptimizedNewsFetcher  # ✅ ИСПРАВЛЕНО
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger
from models.trader_model import NEWS_ENCODED_DIM

from models.trader_model import trader_model_instance
from utils.lot_validator import LotValidator

logger = None
# NEWS_ENCODED_DIM = 128

class SmartPortfolioBroker:
    """Умный брокер с интеграцией всех торговых ядер"""

    def __init__(self, settings: Dict, scheduler: TradingScheduler = None):
        self.rl_config = None
        self.settings = settings
        self.moex = MoexFetcher()

        from core.core_technical_trader import TechnicalTraderCore
        self.technical_core = TechnicalTraderCore()

        # ✅ используем переданный scheduler или создаем новый
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = TradingScheduler()

        # ✅ используем OptimizedNewsFetcher вместо RSSFetcher и NewsTraderCore
        self.news_fetcher = OptimizedNewsFetcher("config/rss_sources.json")
        self.news_fetcher.get_last_news(limit=100)  # Прогрев кэша

        self.risk_manager = RiskManager()
        self.auction_mode = False
        self.portfolio = PortfolioManager()
        self.model = trader_model_instance
        self.model.market_period = "closed"

        # ✅ ЗАГРУЗКА НАСТРОЕК Т-БАНКА
        moex_config = settings.get('moex_schedule', {})
        commission_config = moex_config.get('commission', {})
        self.tbank_config = commission_config.get('tbank', {})

        # Параметры с значениями по умолчанию
        self.tbank_check_start = self.tbank_config.get('check_start_hour', 9)
        self.tbank_check_end = self.tbank_config.get('check_end_hour', 14)
        self.tbank_check_interval = self.tbank_config.get('check_interval_cycles', 60)
        self.tbank_settlement_time = self.tbank_config.get('settlement_time', "14:00")

        global logger
        # Инициализируем глобальный логгер с настройками
        logger = get_logger('SMART_BROKER')
        logger.info(f"[SmartBroker] NewsFetcher инициализирован, статистика: {self.news_fetcher.stats}")

        # ========== ЗАГРУЗКА КОНФИГОВ (ДОЛЖНА БЫТЬ ПЕРВОЙ) ==========
        self.profit_config = settings.get("profit_optimization", {})
        self.rl_config = self._load_rl_config()
        # =============================================================

        # ✅ ЗАГРУЖАЕМ КОНФИГИ СЕНТИМЕНТА
        self.sentiment_config = self.rl_config.get("sentiment_integration", {})
        self.market_sentiment_weight = self.sentiment_config.get("market_sentiment_weight", 0.3)
        self.ticker_sentiment_weight = self.sentiment_config.get("ticker_sentiment_weight", 0.4)
        self.reward_sentiment_bonus = self.sentiment_config.get("reward_sentiment_bonus", 0.5)

        # Загрузка конфигурации действий
        self.action_mapping = self.rl_config.get('action_mapping', {
            "0": "BUY_MIN", "1": "BUY_SMALL", "2": "BUY_NORMAL",
            "3": "HOLD", "4": "SELL_SMALL", "5": "SELL_NORMAL", "6": "SELL_ALL"
        })
        self.position_sizes = self.rl_config.get('position_sizes', {
            "BUY_MIN": 0.02, "BUY_SMALL": 0.05, "BUY_NORMAL": 0.10,
            "SELL_SMALL": 0.25, "SELL_NORMAL": 0.50, "SELL_ALL": 1.0
        })

        # ✅ ИНИЦИАЛИЗАЦИЯ TRAINER (исправление ошибки)
        self.trainer = None
        if settings.get("enable_background_training", True):
            from models.trainer import model_trainer_instance
            self.trainer = model_trainer_instance
            if self.trainer is not None:
                self.trainer.start_background_training()

        # ✅ ЗАГРУЗКА КОНСТАНТ ДЛЯ ПРИБЫЛИ
        self.extreme_pnl_threshold = self.profit_config.get("extreme_pnl_threshold", 0.08)
        self.max_reward = self.profit_config.get("max_reward", 1000.0)
        self.min_reward = self.profit_config.get("min_reward", -1000.0)
        self.reward_scaling_profit = self.profit_config.get("reward_scaling_profit", 100.0)
        self.fast_learning_cycles = self.profit_config.get("fast_learning_cycles", 5)
        self.strategy_adaptation_cycles = self.profit_config.get("strategy_adaptation_cycles", 20)

        # Торговые переменные
        self.last_cycle_time = 0
        self.cycle_count = 0
        self.trading_enabled = True
        self.signals_cache = []
        self.current_tickers = []
        self.ticker_states = {}
        self.pending_experiences = []
        self.strategy_tracker = defaultdict(list)
        self.strategy_usage_counter = defaultdict(int)
        self.last_trade_time = defaultdict(float)

        # Запуск компонентов
        self._initialize_components()

        logger.info(f"Инициализирован. Капитал: {self.portfolio.cash:,.0f}₽")
        logger.info(f"Модель sentiment: {self.model.market_sentiment:.3f}")
        logger.info(f"Макс. позиций: {settings['max_positions']}")
        logger.info(f"Конфиг прибыли загружен: {len(self.profit_config) > 0}")
        logger.info(f"RL конфиг загружен: {len(self.rl_config) > 0}")


        # ✅ Добавляем статистику новостного фетчера
        logger.info(f"NewsFetcher: {self.news_fetcher.stats}")

        self.ticker_states = {}  # {ticker: last_state}
        self.pending_experiences = []  # Опыты для обучения
        self.strategy_tracker = defaultdict(list)  # История стратегий по тикерам

    def _allocate_by_time_horizon(self, signals: List[Dict]) -> Dict[str, List[Dict]]:
        horizons = self.settings.get('time_horizons', {})

        allocation = {
            'day_session': [],
            'three_days': [],
            'week': []
        }

        if not signals:
            return allocation

        for signal in signals:
            ticker = signal['ticker']
            strategy_name = signal.get('strategy', 'balanced')  # ← стратегия ИЗ СИГНАЛА!

            if ticker in self.ticker_states:
                state = self.ticker_states[ticker]  # 151

                # 🔥 ИСПОЛЬЗУЕМ РЕАЛЬНУЮ СТРАТЕГИЮ ИЗ СИГНАЛА!
                strategy_params = self.model.strategies.get(strategy_name, self.model.strategies['balanced'])
                full_state = self.model._create_strategy_state(state, strategy_params)  # 151 → 157

                with torch.no_grad():
                    _, _, price_pred = self.model.policy_net(full_state.unsqueeze(0))
                    pred_probs = self.model.get_price_pred_probs(price_pred)

                    if pred_probs[2] > 0.6:
                        horizon = 'day_session'
                    elif pred_probs[2] > 0.4 or pred_probs[1] > 0.5:
                        horizon = 'three_days'
                    else:
                        horizon = 'week'
            else:
                # Для новых тикеров - по target_hold_time
                strategy_params = self.model.strategies.get(strategy_name, self.model.strategies['balanced'])
                hold_time = strategy_params.get('target_hold_time_hours', 6)

                if hold_time <= 6:
                    horizon = 'day_session'
                elif hold_time <= 24:
                    horizon = 'three_days'
                else:
                    horizon = 'week'

            signal['horizon'] = horizon
            signal['assigned_horizon'] = horizon

            if horizon in allocation:
                allocation[horizon].append(signal)
            else:
                logger.warning(f"Неизвестный горизонт {horizon} для {ticker}, использую week")
                allocation['week'].append(signal)

        # Ограничиваем количество сигналов согласно весам
        for horizon, config in horizons.items():
            if horizon in allocation:
                max_signals = int(len(signals) * config.get('weight', 0.33))
                allocation[horizon] = allocation[horizon][:max_signals]
                logger.debug(f"Горизонт {horizon}: {len(allocation[horizon])} сигналов (max={max_signals})")

        return allocation


    def _load_rl_config(self) -> Dict:
        """Загрузка RL конфига"""
        try:
            with open("config/rl_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить RL конфиг: {e}")
            return {}

    def _get_ticker_sentiment(self, ticker: str) -> float:
        """Получение сентимента для тикера из оптимизированного фетчера"""
        try:
            # Получаем keywords из конфига
            ticker_names = self.rl_config.get('ticker_names', {})
            keywords = ticker_names.get(ticker, [])

            # Ищем новости с keywords
            news_items = self.news_fetcher.search_news(ticker=ticker, limit=5, keywords=keywords)

            if not news_items:
                return 0.0

            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            sentiments = [n.get('sentiment', 0.0) for n in news_with_sentiment]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            if abs(avg_sentiment) > 0.1:
                logger.debug(f"Новостной сентимент для {ticker}: {avg_sentiment:+.3f} ({len(sentiments)} нов.)")

            return avg_sentiment

        except Exception as e:
            logger.error(f"Ошибка получения сентимента для {ticker}: {e}")
            return 0.0

    def _get_market_sentiment(self) -> float:
        """Получение рыночного сентимента из оптимизированного фетчера"""
        try:
            # Получаем все свежие новости
            news_items = self.news_fetcher.get_last_news(limit=50)

            if not news_items:
                return 0.0

            # Анализируем сентимент
            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            # Усредняем сентимент
            sentiments = [n.get('sentiment', 0.0) for n in news_with_sentiment]
            return sum(sentiments) / len(sentiments) if sentiments else 0.0

        except Exception as e:
            logger.error(f"Ошибка получения рыночного сентимента: {e}")
            return 0.0

    def _generate_news_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """Генерация сигналов на основе новостей с маппингом тикер→название"""
        signals = []

        try:
            news_items = self.news_fetcher.get_last_news(limit=100)

            if not news_items:
                return signals

            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            securities = self.moex.get_all_securities()

            # Строим маппинг: тикер → список ключевых слов
            ticker_keywords = {}
            for ticker in prices.keys():
                keywords = set()
                keywords.add(ticker.lower())

                sec_info = securities.get(ticker, {})
                name = sec_info.get('name', '')
                full_name = sec_info.get('full_name', '')

                if name:
                    for word in name.lower().split():
                        clean_word = word.strip('"\'.,;:()[]{}')
                        if len(clean_word) > 2:
                            keywords.add(clean_word)

                if full_name and full_name != name:
                    for word in full_name.lower().split():
                        clean_word = word.strip('"\'.,;:()[]{}')
                        if len(clean_word) > 2:
                            keywords.add(clean_word)

                ticker_config = self.rl_config.get('ticker_names', {})
                extra_names = ticker_config.get(ticker, [])
                for extra in extra_names:
                    keywords.add(extra.lower())

                ticker_keywords[ticker] = keywords

            # Группируем сентимент по тикерам
            ticker_sentiments = defaultdict(list)

            for news in news_with_sentiment:
                sentiment = news.get('sentiment', 0.0)
                title = news.get('title', '').lower()
                summary = news.get('summary', '').lower()
                text = title + ' ' + summary

                for ticker, keywords in ticker_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            ticker_sentiments[ticker].append(sentiment)
                            break

            # Генерируем сигналы
            sentiment_config = self.rl_config.get('sentiment_integration', {})
            sentiment_threshold = sentiment_config.get('ticker_sentiment_weight', 0.3)

            for ticker, price in prices.items():
                sentiments = ticker_sentiments.get(ticker, [])

                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)

                    if abs(avg_sentiment) > sentiment_threshold:
                        signals.append({
                            'ticker': ticker,
                            'action': 'BUY' if avg_sentiment > 0 else 'SELL',
                            'confidence': min(abs(avg_sentiment), 1.0),
                            'price': price,
                            'reason': 'news_analysis',
                            'sentiment': avg_sentiment,
                            'news_count': len(sentiments)
                        })

            logger.debug(f"Сгенерировано новостных сигналов: {len(signals)}")

        except Exception as e:
            logger.error(f"Ошибка генерации новостных сигналов: {e}")

        return signals

    def _execute_trading_decisions(self, signals: List[Dict],
                                   prices: Dict[str, float], securities: Dict):
        """Исполнение торговых решений с RL и учетом лотности"""

        allocated_signals = self._allocate_by_time_horizon(signals)

        all_signals_with_horizon = []
        for horizon, sig_list in allocated_signals.items():
            for sig in sig_list:
                sig['assigned_horizon'] = horizon
                all_signals_with_horizon.append(sig)

        logger.debug(f"[DEBUG] Исполнение {len(signals)} сигналов")

        executed_count = 0

        cooldown_seconds = self.settings.get('cooldown_seconds', 7200)

        for signal in all_signals_with_horizon[:5]:
            ticker = signal['ticker']
            confidence = signal['confidence']
            horizon = signal.get('assigned_horizon', 'balanced')

            if ticker not in prices or ticker not in securities:
                continue

            security_info = securities[ticker]
            lot_size = security_info.get('lot_size', 1)
            min_step = security_info.get('min_step', 0.01)
            price = prices[ticker]

            if min_step and min_step > 0:
                remainder = price % min_step
                if abs(remainder) > 0.0001:
                    price = round(price / min_step) * min_step

            if ticker in self.ticker_states:
                current_state = self.ticker_states[ticker]
            else:
                current_state = self._create_initial_state(ticker, price, security_info)
                self.ticker_states[ticker] = current_state

            strategy, stop_loss, take_profit, action_idx = self._select_buy_strategy_with_action(
                ticker, price, confidence, current_state,
                assigned_horizon=signal.get('assigned_horizon', 'week')
            )

            action_str = self.action_mapping.get(str(action_idx), 'HOLD')

            logger.debug(f"[DEBUG] Сигнал: {ticker} {action_str} (action_idx={action_idx}) conf={confidence:.2f}")

            # ===== HOLD действия (индексы 0, 1, 2) =====
            if action_str.startswith('HOLD'):
                # HOLD не требует исполнения — просто записываем опыт
                ticker_sentiment = self._get_ticker_sentiment(ticker)
                sentiment_data = {
                    'sentiment': ticker_sentiment,
                    'news_count': signal.get('news_count', 0)
                }

                if ticker in self.portfolio.positions:
                    pos = self.portfolio.positions[ticker]
                    if ticker in self.ticker_states:
                        hold_time = (time.time() - pos.get('buy_time', time.time())) / 3600
                        hold_reward = self._calculate_reward(0, hold_time, pos.get('strategy', 'balanced'))

                        current_state_hold = self.ticker_states[ticker]
                        next_state_hold = self._create_next_state(ticker, price)
                        strategy_params = self.model.strategies.get(
                            pos.get('strategy', 'balanced'),
                            self.model.strategies.get('balanced', {})
                        )
                        full_current = self.model._create_strategy_state(current_state_hold, strategy_params)
                        full_next = self.model._create_strategy_state(next_state_hold, strategy_params)

                        self.model.remember_experience(
                            state=full_current,
                            action=action_idx,
                            reward=hold_reward,
                            next_state=full_next,
                            done=False,
                            pnl_rub=0
                        )

                continue

            # ===== ПРОВЕРКА КУЛДАУНА =====
            now = time.time()
            last_trade = self.last_trade_time.get(ticker, 0)
            if now - last_trade < cooldown_seconds:
                logger.debug(f"Кулдаун для {ticker}: прошло {(now - last_trade):.0f}с < {cooldown_seconds}с")
                continue

            # BUY действия
            if action_str.startswith('BUY'):
                atr = security_info.get('atr', None)
                if atr is None:
                    indicators = self.technical_core.calculate_indicators(ticker)
                    atr = indicators.get('atr', None)
                sector = security_info.get('sector', None)

                quantity, actual_risk = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    price=price,
                    stop_loss=stop_loss,
                    atr=atr,
                    confidence=confidence,
                    adv=security_info.get('volume'),
                    sector=sector,
                    lot_size=lot_size
                )

                if quantity <= 0:
                    logger.warning(f"Risk Manager отклонил сделку {ticker}: quantity={quantity}")
                    continue

                available_cash = self.portfolio.cash - self.portfolio.reserved_cash
                total_cost = quantity * price

                if available_cash < total_cost:
                    logger.warning(f"❌ Недостаточно средств для {ticker}: нужно {total_cost:.2f}₽")
                    continue

                daily_volume_rub = security_info.get('volume') or 0
                liquidity_multiplier = self.settings.get('liquidity_check_multiplier', 10.0)
                if daily_volume_rub > 0 and daily_volume_rub < total_cost * liquidity_multiplier:
                    logger.warning(f"⚠️ Низкая ликвидность {ticker}")
                    continue

                if self.portfolio.buy(ticker, quantity, price, strategy,
                                      lot_size=lot_size, min_step=min_step,
                                      stop_loss=stop_loss, take_profit=take_profit,
                                      time_horizon=horizon):
                    self.last_trade_time[ticker] = now

                    ticker_sentiment = self._get_ticker_sentiment(ticker)
                    sentiment_data = {
                        'sentiment': ticker_sentiment,
                        'news_count': signal.get('news_count', 0)
                    }

                    self._record_rl_experience(
                        ticker=ticker, state=current_state, action=action_idx,
                        strategy=strategy, price=price, quantity=quantity,
                        sentiment_data=sentiment_data
                    )

                    executed_count += 1
                    logger.info(
                        f"✅ BUY {ticker}: {quantity} @ {price:.2f} "
                        f"(risk={actual_risk:.0f}₽, strategy={strategy})"
                    )

            # SELL действия
            elif action_str.startswith('SELL') and ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                sell_ratio = self.position_sizes.get(action_str, 0.5)

                qty = int(pos['qty'] * sell_ratio)
                pos_lot_size = pos.get('lot_size', 1)

                if pos_lot_size > 1:
                    qty = (qty // pos_lot_size) * pos_lot_size

                if qty <= 0:
                    continue

                pos_info = {
                    'avg_price': pos['avg_price'],
                    'strategy': pos.get('strategy', 'balanced'),
                    'qty': pos['qty'],
                    'buy_time': pos.get('buy_time', time.time())
                }

                success, pnl = self.portfolio.sell(ticker, qty, price)

                if success:
                    self.last_trade_time[ticker] = now

                    self._complete_rl_experience(ticker, price, actual_pnl=pnl, pos_info=pos_info)

                    ticker_sentiment = self._get_ticker_sentiment(ticker)
                    next_state = self._create_next_state(ticker, price)
                    strategy_params = self.model.strategies.get(
                        pos_info.get('strategy', 'balanced'),
                        self.model.strategies.get('balanced', {})
                    )
                    full_current = self.model._create_strategy_state(current_state, strategy_params)
                    full_next = self.model._create_strategy_state(next_state, strategy_params)

                    hold_time = (time.time() - pos_info.get('buy_time', time.time())) / 3600
                    reward = self._calculate_reward(pnl, hold_time, pos_info.get('strategy', 'balanced'))

                    self.model.remember_experience(
                        state=full_current,
                        action=action_idx,
                        reward=reward,
                        next_state=full_next,
                        done=True,
                        pnl_rub=pnl
                    )

                    executed_count += 1
                    logger.info(
                        f"💰 SELL {ticker}: {qty} @ {price:.2f} "
                        f"({sell_ratio * 100:.0f}% позиции, PnL: {pnl:+.2f}₽)"
                    )

                    if hasattr(self.model, 'record_strategy_outcome'):
                        self.model.record_strategy_outcome(
                            strategy_name=pos.get('strategy', 'balanced'),
                            action='SELL',
                            pnl=pnl,
                            hold_time=time.time() - pos.get('buy_time', time.time())
                        )

        logger.debug(f"[DEBUG] Всего исполнено сделок: {executed_count}")
        return executed_count

    def _record_rl_experience(self, ticker: str, state: torch.Tensor,
                              action: int, strategy: str, price: float,
                              quantity: int, sentiment_data: Dict = None):
        """Запись опыта в память"""

        # Проверка валидности action
        action_dim = self.rl_config.get('action_dim', 7)
        if action >= action_dim:
            logger.error(f"❌ Некорректный action={action} (max={action_dim - 1}), заменяю на HOLD=3")
            action = 3

        logger.info(f"📝 Запись опыта: {ticker} action={action} strategy={strategy}")

        strategy_params = self.model.strategies.get(strategy, self.model.strategies.get('balanced', {}))
        full_state = self.model._create_strategy_state(state, strategy_params)

        experience = {
            'ticker': ticker,
            'start_state': full_state.cpu(),
            'action': action,
            'strategy': strategy,
            'entry_price': price,
            'entry_time': time.time(),
            'quantity': quantity,
            'sentiment_data': sentiment_data,
            'completed': False,
            'is_priority': self._is_priority_experience(sentiment_data)
        }
        self.pending_experiences.append(experience)

        logger.debug(f"✅ pending_experiences: {len(self.pending_experiences)}")

    def _is_priority_experience(self, sentiment_data: Optional[Dict]) -> bool:
        """Определение приоритетности опыта через конфиги"""
        if not sentiment_data:
            return False

        strategy_config = self.model.strategy_config
        priority_config = strategy_config.get("strategy_selection", {}).get("priority_learning", {})

        if not priority_config.get("enable_priority_training", True):
            return False

        sentiment_threshold = priority_config.get("sentiment_threshold", 0.6)
        news_count_threshold = priority_config.get("news_count_threshold", 3)

        sentiment_strength = abs(sentiment_data.get('sentiment', 0))
        impact_level = sentiment_data.get('impact_level', 'low_impact')
        news_count = sentiment_data.get('news_count', 0)

        is_priority = (
                sentiment_strength > sentiment_threshold or
                impact_level in priority_config.get("impact_levels", ["high_impact"]) or
                news_count >= news_count_threshold
        )

        return is_priority

    def _complete_rl_experience(self, ticker: str, exit_price: float, actual_pnl: float = None, pos_info: dict = None):
        """Завершение опыта и запись в память модели"""

        logger.info(f"🏁 Завершение опыта: {ticker} PnL={actual_pnl}")

        for exp in self.pending_experiences:
            if exp['ticker'] == ticker and not exp.get('completed', False):

                pnl = actual_pnl if actual_pnl is not None else 0.0

                next_base_state = self._create_next_state(ticker, exit_price)
                strategy_params = self.model.strategies.get(exp['strategy'], self.model.strategies.get('balanced', {}))

                full_start_state = self.model._create_strategy_state(
                    exp['start_state'].to(self.model.device), strategy_params
                )
                full_next_state = self.model._create_strategy_state(next_base_state, strategy_params)

                hold_time = (time.time() - exp['entry_time']) / 3600
                reward = self._calculate_reward(pnl, hold_time, exp['strategy'])

                logger.info(f"📊 Reward для {ticker}: {reward:.4f} (PnL={pnl:.2f}₽, hold={hold_time:.1f}ч)")

                self.model.remember_experience(
                    state=full_start_state,
                    action=exp['action'],
                    reward=reward,
                    next_state=full_next_state,
                    done=True,
                    pnl_rub=pnl
                )

                exp['completed'] = True

                # Обновление статистики тикера
                if hasattr(self.model, 'ticker_stats'):
                    stats = self.model.ticker_stats[ticker]
                    stats['total_trades'] += 1
                    stats['total_pnl'] += pnl
                    if pnl > 0:
                        stats['profitable_trades'] += 1
                    stats['success_rate'] = stats['profitable_trades'] / max(stats['total_trades'], 1)
                    stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['total_trades'] - 1) + hold_time) / stats[
                        'total_trades']
                    logger.info(
                        f"📈 Статистика {ticker}: trades={stats['total_trades']}, win_rate={stats['success_rate']:.1%}")

        # Очистка завершенных
        self.pending_experiences = [e for e in self.pending_experiences if not e.get('completed', False)]

    def _calculate_reward(self, pnl: float, hold_time: float, strategy: str) -> float:
        """Расчёт награды для RL - ВСЕ ПАРАМЕТРЫ ИЗ КОНФИГА"""

        portfolio_value = self.portfolio.get_total_value({})
        if portfolio_value <= 0:
            portfolio_value = self.settings.get('initial_capital_rub', 10000)

        # Загружаем конфиг наград
        reward_config = self.rl_config.get('reward_config', {})

        # 1. Нормируем на капитал (доходность в долях)
        pnl_percent = pnl / portfolio_value if portfolio_value > 0 else 0.0

        # 2. Масштабируем (коэффициент из конфига)
        scale = reward_config.get('pnl_scale_factor', 100.0)
        reward = pnl_percent * scale

        # 3. Бонус за положительный PnL (параметры из конфига)
        profit_bonus_enabled = reward_config.get('profit_bonus_enabled', True)
        if profit_bonus_enabled and pnl_percent > 0:
            fixed_bonus = reward_config.get('profit_fixed_bonus', 0.5)
            proportional_bonus_scale = reward_config.get('profit_proportional_bonus_scale', 200.0)
            reward += fixed_bonus
            reward += pnl_percent * proportional_bonus_scale
        elif pnl_percent < 0:
            # Меньший штраф за убыток
            loss_penalty_scale = reward_config.get('loss_penalty_scale', 50.0)
            reward += pnl_percent * loss_penalty_scale

        # 4. Бонус за скорость (только для прибыльных сделок)
        if reward_config.get('speed_bonus_enabled', True) and pnl_percent > 0:
            strategy_params = self.model.strategies.get(strategy, self.model.strategies['balanced'])
            target_time = strategy_params.get('target_hold_time_hours', 6)
            if hold_time < target_time:
                speed_max = reward_config.get('speed_bonus_max_percent', 50.0)
                speed_bonus = pnl_percent * speed_max * (1 - hold_time / target_time)
                reward += speed_bonus

        # 5. Штраф за долгий убыток или долгий HOLD
        if reward_config.get('time_penalty_enabled', True):
            if pnl_percent < 0 and hold_time > reward_config.get('loss_hold_threshold', 6.0):
                time_penalty_max = reward_config.get('time_penalty_max_percent', 20.0)
                time_penalty = abs(pnl_percent) * time_penalty_max
                reward -= time_penalty
            elif pnl_percent <= 0 and hold_time > reward_config.get('hold_penalty_threshold', 24.0):
                hold_penalty_rate = reward_config.get('hold_penalty_rate', 0.1)
                reward -= hold_penalty_rate * (hold_time / reward_config.get('hold_penalty_threshold', 24.0))

        # 6. Штраф за комиссию (относительный)
        if reward_config.get('commission_penalty_enabled', True):
            total_commission = getattr(self.portfolio, 'total_commission', 0.0)
            if total_commission > 0 and portfolio_value > 0:
                commission_ratio = total_commission / portfolio_value
                penalty_scale = reward_config.get('commission_penalty_scale', 50.0)
                reward -= commission_ratio * penalty_scale

        # 7. Штраф за концентрацию
        if reward_config.get('concentration_penalty_enabled', True):
            positions_count = len(self.portfolio.positions)
            max_positions = reward_config.get('max_positions_before_penalty', 5)
            if positions_count > max_positions:
                penalty_per_pos = reward_config.get('concentration_penalty_per_position', 0.1)
                reward -= penalty_per_pos * (positions_count - max_positions)

        # 8. Обрезание награды (из конфига)
        max_reward = reward_config.get('reward_clip_max', 20.0)
        min_reward = reward_config.get('reward_clip_min', -10.0)
        reward = max(min_reward, min(max_reward, reward))

        return reward

    def _create_initial_state(self, ticker: str, price: float, security_info: Dict) -> torch.Tensor:
        """Создание начального состояния для RL"""
        momentum = security_info.get('momentum', 0.0)
        sentiment = self._get_ticker_sentiment(ticker)

        indicators = self.technical_core.calculate_indicators(ticker)

        ticker_names = self.rl_config.get('ticker_names', {})
        keywords = ticker_names.get(ticker, [])
        news_items = self.news_fetcher.search_news(ticker=ticker, limit=3, keywords=keywords)
        news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
        news_features = self.model.encode_news(news_texts)

        market_sentiment = self._get_market_sentiment()
        macro_data = self.moex.get_macro_data()

        # Коэффициенты нормализации из конфига модели
        norm = self.model.normalization

        enhanced_market_data = {
            'volume': indicators.get('volume', 0),
            'spread': security_info.get('spread', 0.01),
            'rsi': indicators.get('rsi', 50),
            'volatility': indicators.get('atr', 0) / price if price > 0 else 0.1,
            'sma_10_ratio': indicators.get('sma_10', price) / price if price > 0 else 1.0,
            'sma_20_ratio': indicators.get('sma_20', price) / price if price > 0 else 1.0,
            'bb_position': indicators.get('bb_position', 0.5),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'atr': indicators.get('atr', 0),
            'market_cap': security_info.get('market_cap', 0),
            'lot_size': security_info.get('lot_size', 1),
            'min_step': security_info.get('min_step', 0.01),
            'sector': security_info.get('sector', 'other'),
            'momentum': momentum,
            'imoex': macro_data.get('imoex', 0),
            'imoex_change': macro_data.get('imoex_change', 0),
            'rtsi': macro_data.get('rtsi', 0),
            'rtsi_change': macro_data.get('rtsi_change', 0),
            'rvi': macro_data.get('rvi', 20.0),
            'rvi_change': macro_data.get('rvi_change', 0),
            'moexog': macro_data.get('moexog', 0),
            'moexfn': macro_data.get('moexfn', 0),
            'brent': macro_data.get('brent', 0),
            'brent_change': macro_data.get('brent_change', 0),
            'market_liquidity_ratio': macro_data.get('market_liquidity_ratio', 0.0),
            'market_activity_score': macro_data.get('market_activity_score', 0.0),
            # MARKET_FEATURES — нормализация из конфига
            'spread_pct': (security_info.get('spread', 0) / price) if price > 0 else 0.0,
            'market_mood': macro_data.get('market_mood', 0.0),
            'shares_turnover': macro_data.get('shares_turnover', 0) / norm.get('shares_turnover_divisor', 1e12),
            'rvi_normalized': macro_data.get('rvi', 20.0) / norm.get('rvi_divisor', 100.0),
            'imoex_normalized': macro_data.get('imoex', 0) / norm.get('imoex_divisor', 4000.0),
            'market_cap_total': macro_data.get('market_cap', 0) / norm.get('market_cap_divisor_total', 1e14),
            'liquidity_ratio': macro_data.get('market_liquidity_ratio', 0.0),
            'cbr_rate_normalized': macro_data.get('cbr_rate', 0.0) / norm.get('cbr_rate_divisor', 20.0),
            'vix': macro_data.get('vix', 0.0) / norm.get('vix_divisor', 50.0),
            'moexog_normalized': macro_data.get('moexog', 0) / norm.get('moexog_divisor', 10000.0),
        }

        state = self.model.build_state_vector(
            ticker=ticker,
            price=price,
            momentum=momentum,
            sentiment=sentiment,
            news_features=news_features,
            market_data=enhanced_market_data,
            market_sentiment=market_sentiment,
            portfolio=self.portfolio
        )

        return state

    def _select_buy_strategy_with_action(self, ticker: str, price: float,
                                         confidence: float, base_state: torch.Tensor,
                                         assigned_horizon: str = 'week') -> Tuple[str, float, float, int]:
        """Выбор стратегии и действия с таймаутом"""

        timeout = self.settings.get('strategy_selection_timeout', 10)

        def _impl():
            if base_state.dtype != torch.float32:
                base_state_local = base_state.to(dtype=torch.float32)
            else:
                base_state_local = base_state

            sentiment_score = self._get_ticker_sentiment(ticker)

            market_context = {
                'market_sentiment': self.model.market_sentiment,
                'volatility': self.model.volatility_index,
                'confidence': confidence,
                'time_of_day': datetime.now().hour / 24.0,
                'ticker_sentiment': sentiment_score,
                'assigned_horizon': assigned_horizon,
            }

            action, final_strategy, strategy_confidence = self.model.choose_action_with_strategy(
                state=base_state_local,
                ticker=ticker,
                price=price,
                market_context=market_context
            )
            return action, final_strategy, strategy_confidence, sentiment_score

        result_queue = queue.Queue()
        thread = threading.Thread(target=lambda: result_queue.put(_impl()), daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.error(f"⏰ ТАЙМАУТ выбора стратегии для {ticker}, использую balanced")
            action = 3  # HOLD
            final_strategy = 'balanced'
            strategy_confidence = 0.5
            sentiment_score = 0.0
        else:
            try:
                action, final_strategy, strategy_confidence, sentiment_score = result_queue.get_nowait()
            except:
                action = 3  # HOLD
                final_strategy = 'balanced'
                strategy_confidence = 0.5
                sentiment_score = 0.0

        strategy_config = self.model.strategies.get(final_strategy, self.model.strategies['balanced'])
        base_stop_loss = strategy_config.get('stop_loss_percent', 2.5)
        base_take_profit = strategy_config.get('take_profit_percent', 5.0)

        volatility_factor = 1.0
        if hasattr(self.model, 'volatility_index'):
            volatility_factor = 1.0 + self.model.volatility_index

        confidence_factor = 0.5 + strategy_confidence
        adjusted_stop_loss = base_stop_loss * volatility_factor / confidence_factor
        adjusted_take_profit = base_take_profit * volatility_factor * confidence_factor

        adjusted_stop_loss = max(0.5, min(10.0, adjusted_stop_loss))
        adjusted_take_profit = max(1.0, min(20.0, adjusted_take_profit))

        stop_loss = price * (1 - adjusted_stop_loss / 100)
        take_profit = price * (1 + adjusted_take_profit / 100)

        logger.debug(
            f"🤖 {ticker}: action={action}, strategy={final_strategy}, "
            f"conf={strategy_confidence:.2f}, SL={adjusted_stop_loss:.1f}%, TP={adjusted_take_profit:.1f}%"
        )

        return final_strategy, stop_loss, take_profit, action

    def _periodic_learning(self):
        """ОПТИМИЗИРОВАННОЕ онлайн-обучение с приоритетами"""
        try:
            # 🔥 ДИАГНОСТИКА
            print(f"\n📊 ДИАГНОСТИКА ПАМЯТИ:")
            print(f"   memory size: {len(self.model.memory)}")
            if hasattr(self.model, 'prioritized_buffer'):
                print(f"   prioritized_buffer size: {self.model.prioritized_buffer.size}")

            # ✅ ИСПРАВЛЕНО: определяем переменную!
            enable_extreme = self.profit_config.get("enable_extreme_learning", True)

            # 1. ПРИОРИТЕТНОЕ ОБУЧЕНИЕ
            if hasattr(self.model, 'learn_from_prioritized'):
                if self.model.prioritized_buffer.size >= 32:
                    priority_loss = self.model.learn_from_prioritized(batch_size=32)
                    if priority_loss:
                        logger.debug(f"Приоритетное обучение: Loss={priority_loss:.6f}")
                else:
                    print(f"   ⚠️ Недостаточно опытов в prioritized_buffer: {self.model.prioritized_buffer.size} < 32")

            # 2. Критические сделки
            critical_trades = []
            for ticker, pos in self.portfolio.positions.items():
                current_price = self.moex.get_price(ticker)
                if not current_price:
                    continue

                entry_price = pos.get('avg_price', 0)
                if entry_price <= 0:
                    continue

                pnl_pct = (current_price - entry_price) / entry_price * 100

                if abs(pnl_pct) > self.extreme_pnl_threshold * 100:
                    critical_trades.append({
                        'ticker': ticker,
                        'pnl_pct': pnl_pct,
                        'is_profit': pnl_pct > 0
                    })

            # 3. Обучение на критических сделках
            if enable_extreme and critical_trades and len(self.model.memory) > 32:
                extreme_batch = self._create_extreme_batch(critical_trades)
                if extreme_batch:
                    loss = self._train_extreme(extreme_batch)
                    if loss:
                        profit_count = sum(1 for t in critical_trades if t['is_profit'])
                        loss_count = len(critical_trades) - profit_count
                        logger.info(f"[ЭКСТРЕМ-обучение] {len(critical_trades)} сделок "
                                    f"(прибыль: {profit_count}, убытки: {loss_count}) Loss={loss:.6f}")

            # 4. Регулярное обучение
            if self.cycle_count % (self.fast_learning_cycles * 2) == 0:
                if len(self.model.memory) >= 32:
                    regular_loss = self.model.learn_from_experience(batch_size=32)
                    if regular_loss:
                        logger.debug(f"Регулярное обучение: Loss={regular_loss:.6f}")

            # 5. Адаптация стратегий
            if self.cycle_count % self.strategy_adaptation_cycles == 0:
                self._adapt_strategies_for_profit()

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _train_extreme(self, extreme_batch):
        """АГРЕССИВНОЕ обучение на экстремальных сделках"""
        try:
            if not extreme_batch:
                return None

            learning_config = self.rl_config.get("learning", {})
            aggressive_multiplier = learning_config.get("aggressive_lr_multiplier", 3.0)
            training_steps = learning_config.get("extreme_training_steps", 3)

            # ВРЕМЕННО увеличиваем learning rate
            original_lr = self.model.policy_optimizer.param_groups[0]['lr']
            self.model.policy_optimizer.param_groups[0]['lr'] = original_lr * aggressive_multiplier

            try:
                # Безопасная подготовка батча
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

                for exp in extreme_batch:
                    if 'state' in exp and 'action' in exp and 'reward' in exp and 'next_state' in exp and 'done' in exp:
                        states.append(exp['state'])
                        actions.append(exp['action'])
                        rewards.append(exp['reward'])
                        next_states.append(exp['next_state'])
                        dones.append(exp['done'])

                if not states:
                    return None

                # Конвертируем в тензоры
                states_tensor = torch.stack(states).to(self.model.device)
                actions_tensor = torch.LongTensor(actions).to(self.model.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.model.device)
                next_states_tensor = torch.stack(next_states).to(self.model.device)
                dones_tensor = torch.FloatTensor(dones).to(self.model.device)

                # АГРЕССИВНОЕ обучение
                total_loss = 0
                successful_steps = 0

                for step in range(training_steps):
                    loss = self.model.learn_from_experience_custom(
                        states=states_tensor,
                        actions=actions_tensor,
                        rewards=rewards_tensor,
                        next_states=next_states_tensor,
                        dones=dones_tensor
                    )
                    if loss is not None:
                        total_loss += loss
                        successful_steps += 1

                avg_loss = total_loss / successful_steps if successful_steps > 0 else None
                return avg_loss

            finally:
                # Восстанавливаем оригинальный LR
                self.model.policy_optimizer.param_groups[0]['lr'] = original_lr

        except Exception as e:
            logger.error(f"Ошибка экстремального обучения: {e}")
            return None

    def _create_extreme_batch(self, critical_trades, max_size=16):
        """Создание экстремального батча"""
        try:
            extreme_experiences = []
            critical_tickers = {trade['ticker'] for trade in critical_trades}

            learning_config = self.rl_config.get("learning", {})
            max_size = learning_config.get("extreme_batch_size", 16)

            memory_list = list(self.model.memory)

            for exp in reversed(memory_list):
                if len(extreme_experiences) >= max_size:
                    break

                exp_ticker = None
                if 'ticker' in exp:
                    exp_ticker = exp['ticker']
                elif 'market_conditions' in exp and 'ticker' in exp['market_conditions']:
                    exp_ticker = exp['market_conditions']['ticker']
                elif hasattr(exp, 'get') and callable(exp.get):
                    exp_ticker = exp.get('ticker')

                if exp_ticker in critical_tickers:
                    modified_exp = exp.copy() if hasattr(exp, 'copy') else dict(exp)

                    if 'reward' in modified_exp:
                        reward_value = modified_exp['reward']
                        reward_multiplier = 2.0 if 'is_profit' not in modified_exp or modified_exp.get('is_profit') else 2.5

                        if isinstance(reward_value, (int, float)):
                            modified_exp['reward'] = reward_value * reward_multiplier
                        elif isinstance(reward_value, torch.Tensor):
                            modified_exp['reward'] = reward_value.clone() * reward_multiplier
                        else:
                            continue

                    extreme_experiences.append(modified_exp)

            logger.debug(f"Создан экстремальный батч: {len(extreme_experiences)}/{max_size} записей")
            return extreme_experiences

        except Exception as e:
            logger.error(f"Ошибка создания экстремального батча: {e}")
            return []

    def _create_priority_batch(self, critical_trades, max_size=8):
        """Создание приоритетного батча"""
        if not self.model.memory or len(self.model.memory) < 20:
            return []

        recent_memory = list(self.model.memory)[-50:]
        critical_tickers = {trade['ticker'] for trade in critical_trades}
        priority_experiences = []

        for exp in recent_memory:
            exp_ticker = None
            if 'ticker' in exp:
                exp_ticker = exp['ticker']
            elif 'market_conditions' in exp and 'ticker' in exp['market_conditions']:
                exp_ticker = exp['market_conditions']['ticker']

            if exp_ticker in critical_tickers:
                priority_experiences.append(exp)

            if len(priority_experiences) >= max_size:
                break

        return priority_experiences

    def _train_online_priority(self, priority_batch, batch_size=8):
        """Быстрое онлайн-обучение на приоритетном батче"""
        try:
            states = torch.stack([exp['state'] for exp in priority_batch]).to(self.model.device)
            actions = torch.LongTensor([exp['action'] for exp in priority_batch]).to(self.model.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in priority_batch]).to(self.model.device)
            next_states = torch.stack([exp['next_state'] for exp in priority_batch]).to(self.model.device)
            dones = torch.FloatTensor([exp['done'] for exp in priority_batch]).to(self.model.device)

            self.model.policy_net.train()

            current_probs, current_values = self.model.policy_net(states)

            with torch.no_grad():
                _, next_values = self.model.policy_net(next_states)

            target_values = rewards + (1 - dones) * self.model.gamma * next_values

            value_loss = torch.nn.SmoothL1Loss()(current_values, target_values.detach())

            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            total_loss = value_loss + policy_loss

            self.model.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 0.5)
            self.model.policy_optimizer.step()

            self.model.policy_net.eval()

            return total_loss.item()

        except Exception as e:
            logger.error(f"Ошибка онлайн-обучения: {e}")
            return None

    def _has_high_impact_news(self, ticker):
        """Проверка наличия высоковлиятельных новостей"""
        try:
            sentiment = self._get_ticker_sentiment(ticker)
            return abs(sentiment) > 0.5
        except:
            return False

    def _evaluate_strategies(self):
        """Оценка эффективности стратегий"""
        strategies_report = {}

        for strategy_name, perf in self.model.strategy_performance.items():
            if perf['total_trades'] > 0:
                strategies_report[strategy_name] = {
                    'total_trades': perf['total_trades'],
                    'win_rate': f"{perf['win_rate'] * 100:.1f}%",
                    'avg_pnl': f"{perf['avg_pnl']:.2f}",
                    'total_pnl': f"{perf['total_pnl']:.2f}"
                }

        if strategies_report:
            logger.info(f"Эффективность стратегий: {json.dumps(strategies_report, indent=2)}")



    def _initialize_components(self):
        """Инициализация всех компонентов"""
        self.portfolio.initial_capital = self.settings.get("initial_capital_rub", 10000)
        self.portfolio.max_positions = self.settings.get("max_positions", 5)

        # ✅ Запуск непрерывного сбора новостей (опционально)
        if self.settings.get("enable_news_core", True):
            # NewsFetcher сам управляет кэшем, запускаем фоновый сбор
            import threading
            def background_fetch():
                while True:
                    try:
                        start_time = time.time()
                        news = self.news_fetcher.get_last_news(limit=100)
                        elapsed = time.time() - start_time

                        if news:
                            logger.debug(f"📰 Фоновый сбор: {len(news)} новостей за {elapsed:.1f}с")
                        else:
                            logger.warning("⚠️ Фоновый сбор: новостей нет")

                    except Exception as e:
                        logger.error(f"❌ Ошибка фонового сбора: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

                    time.sleep(120)  # 2 минуты

            fetch_thread = threading.Thread(target=background_fetch, daemon=True)
            fetch_thread.start()
            logger.info("Запущен фоновый сбор новостей")
            logger.info(f"📊 Текущая статистика NewsFetcher: {self.news_fetcher.stats}")

        # Загрузка состояния портфеля
        self._load_portfolio_state()

        # Запуск планировщика задач
        self.scheduler.schedule_daily_tasks(
            pre_market_callback=self.pre_session_analysis,
            market_open_callback=self.on_market_open,
            market_close_callback=self.on_market_close,
            post_market_callback=self.post_session_analysis,
            clearing_liquidity_callback=self.check_clearing_liquidity,
            z0_deadline_callback=self.execute_z0_deadline,
            clearing_17_callback=self.process_clearing_17,
            clearing_19_callback=self.process_clearing_19,
            commission_callback = self.process_pending_commissions
        )

        # Проверяем текущее состояние рынка при запуске
        current_time = datetime.now(self.scheduler.moscow_tz)
        if self.scheduler.is_trading_time(current_time):
            logger.info("РЫНОК УЖЕ ОТКРЫТ при запуске, активируем режим торговли")
            self.trading_enabled = True
            # Получаем текущие цены для проверки стопов
            prices = self._get_current_prices()
            if prices:
                self.check_stops_and_tp(prices)
        else:
            logger.info("Рынок закрыт при запуске, ожидаем открытия")
            self.trading_enabled = False

        logger.info("SmartBroker: все компоненты инициализированы")

    def _load_portfolio_state(self):
        """Загрузка состояния портфеля из файла"""
        try:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

                self.portfolio.positions = state.get('positions', {})
                self.portfolio.cash = state.get('cash', self.settings["initial_capital_rub"])

                self.portfolio.reserved_cash = state.get('reserved_cash', 0)
                self.portfolio.pending_commissions = state.get('pending_commissions', [])

                for ticker, pos in self.portfolio.positions.items():
                    if 'buy_time' not in pos:
                        pos['buy_time'] = time.time()

                logger.info(f"Загружено состояние портфеля: {len(self.portfolio.positions)} позиций, "
                            f"{self.portfolio.cash:,.0f}₽ кэша")

        except Exception as e:
            logger.warning(f"Не удалось загрузить состояние портфеля: {e}")
            self._save_portfolio_state()

    def _save_portfolio_state(self):
        """Сохранение состояния портфеля"""
        try:
            state = {
                'total_value': self.portfolio.get_total_value({}),
                'cash': self.portfolio.cash,
                'positions': self.portfolio.positions,
                'last_update': datetime.now().isoformat(),
                'initial_capital': self.settings["initial_capital_rub"],
                'reserved_cash': getattr(self.portfolio, 'reserved_cash', 0),
                'pending_commissions': getattr(self.portfolio, 'pending_commissions', []),
                'trade_history': getattr(self.portfolio, 'trade_history', [])[-100:],
                'daily_trades': getattr(self.portfolio, 'daily_trades', []),
                'commission_spent_today': getattr(self.portfolio, 'commission_spent_today', 0.0),
                'total_commission': getattr(self.portfolio, 'total_commission', 0.0),
                'total_trades': getattr(self.portfolio, 'total_trades', 0),
                'total_pnl': getattr(self.portfolio, 'total_pnl', 0.0)
            }

            with open('data/portfolio_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Ошибка сохранения состояния портфеля: {e}")

    def pre_session_analysis(self):
        """Предсессионный анализ"""
        logger.info("=" * 60)
        logger.info("ПРЕДСЕССИОННЫЙ АНАЛИЗ")
        logger.info("=" * 60)

        self.risk_manager.reset_daily_metrics()

        self._set_daily_start_capital()

        # Анализ новостей
        self.news_fetcher.get_last_news(limit=100)

        securities = self.moex.get_all_securities()
        logger.info(f"Загружено бумаг для анализа: {len(securities)}")

        # Обновление рыночного настроения
        market_sentiment = self._get_market_sentiment()
        self.model.update_market_sentiment(market_sentiment)

        logger.info("Предсессионный анализ завершен")

    def on_market_open(self):
        """Действия при открытии рынка"""
        logger.info("РЫНОК ОТКРЫТ")
        self.trading_enabled = True

        prices = self._get_current_prices()
        if prices:
            self.check_stops_and_tp(prices)

    def on_market_close(self):
        """Действия при закрытии рынка"""
        logger.info("РЫНОК ЗАКРЫТ")
        self.trading_enabled = False

        # Фиксация дневной прибыли
        self._fixate_daily_profit()

        self._save_portfolio_state()
        self.model.save_model()

    def post_session_analysis(self):
        """Послерыночный анализ"""
        logger.info("ПОСЛЕРЫНОЧНЫЙ АНАЛИЗ")

        total_value = self.portfolio.get_total_value({})
        initial = self.settings["initial_capital_rub"]
        daily_pnl = total_value - self.portfolio.cash - sum(
            p['qty'] * p['avg_price'] for p in self.portfolio.positions.values()
        )

        logger.info(f"Итоги дня: Капитал {total_value:,.0f}₽, PnL: {daily_pnl:+,.0f}₽")
        logger.info(f"С начала: {((total_value / initial) - 1) * 100:+.2f}%")

        self._generate_daily_report()

    def run_cycle(self):
        """Основной торговый цикл с защитой от зависания"""
        thread = threading.Thread(target=self._run_cycle_impl, daemon=True)
        thread.start()
        thread.join(timeout=self.settings.get('cycle_timeout', 90))

        if thread.is_alive():
            logger.error(f"⏰ ТАЙМАУТ ЦИКЛА #{self.cycle_count + 1} (>90с), принудительно пропускаю")
            self.cycle_count += 1

    def _run_cycle_impl(self):
        """Реализация цикла (вся логика из старого run_cycle)"""
        logger.debug(f"ЦИКЛ #{self.cycle_count} | pending_experiences: {len(self.pending_experiences)} | позиций: {len(self.portfolio.positions)}")

        current_hour = datetime.now().hour
        if (self.tbank_check_start <= current_hour < self.tbank_check_end and
                self.cycle_count % self.tbank_check_interval == 0):
            self.process_pending_commissions()

        if not self.trading_enabled or not self.scheduler.is_trading_time():
            return

        period_info = self.scheduler.can_trade_now()
        self.auction_mode = period_info['current_period'] in ['auction_open', 'auction_close']

        if hasattr(self, 'model') and self.model:
            self.model.market_period = period_info['current_period']

        if self.cycle_count % 10 == 0:
            logger.info(f"Текущий период MOEX: {period_info['current_period']}, "
                        f"аукцион: {self.auction_mode}")

        if not period_info['can_place_orders']:
            logger.debug(f"Торговля недоступна: период {period_info['current_period']}")
            return

        cycle_start = time.time()
        self.cycle_count += 1

        try:
            logger.debug(f"=== Торговый цикл #{self.cycle_count} ===")

            securities = self.moex.get_all_securities()
            if not securities:
                logger.warning("Не удалось получить список бумаг")
                return

            top_n = 120
            tickers = sorted(securities.items(),
                             key=lambda x: x[1].get('volume', 0),
                             reverse=True)[:top_n]
            tickers = [t[0] for t in tickers]
            self.current_tickers = tickers

            prices = {}
            for ticker in tickers:
                price = self.moex.get_price(ticker)
                if price:
                    prices[ticker] = price

            if len(prices) < 10:
                logger.warning(f"Слишком мало цен: {len(prices)}")
                return

            all_signals = []

            if self.settings.get("enable_news_core", True):
                news_signals = self._generate_news_signals(prices)
                all_signals.extend(news_signals)
                logger.debug(f"Сгенерировано новостных сигналов: {len(news_signals)}")

            if self.settings.get("enable_technical_core", True):
                tech_signals = self.technical_core.analyze_all_tickers(prices)
                all_signals.extend(tech_signals)
                logger.debug(f"Сгенерировано технических сигналов: {len(tech_signals)}")

            filtered_signals = self._aggregate_signals(all_signals)

            # Применяем фильтрацию по успешности тикеров
            filtered_signals = self._filter_signals_by_success_rate(filtered_signals)

            self.signals_cache = filtered_signals[:10]
            self.check_stops_and_tp(prices)

            # ===== HOLD REWARD =====
            hold_config = self.rl_config.get('hold_reward', {})
            if hold_config.get('enabled', False):
                hold_interval = hold_config.get('interval_cycles', 5)
                if self.cycle_count % hold_interval == 0:
                    max_positions = hold_config.get('max_positions_per_cycle', 3)

                    # Получаем индексы всех HOLD-действий из конфига
                    action_mapping = self.rl_config.get('action_mapping', {})
                    hold_action_indices = []
                    for key, value in action_mapping.items():
                        if value.startswith('HOLD'):
                            hold_action_indices.append(int(key))
                    if not hold_action_indices:
                        # Fallback: обратная совместимость со старыми конфигами
                        hold_action_indices = [3]

                    # Блок 1: HOLD для открытых позиций
                    for ticker, pos in list(self.portfolio.positions.items())[:max_positions]:
                        if ticker in self.ticker_states:
                            current_price = self.moex.get_price(ticker)
                            if current_price:
                                current_state = self.ticker_states[ticker]
                                next_state = self._create_next_state(ticker, current_price)

                                strategy = pos.get('strategy', 'balanced')
                                strategy_params = self.model.strategies[strategy]

                                full_current = self.model._create_strategy_state(current_state, strategy_params)
                                full_next = self.model._create_strategy_state(next_state, strategy_params)

                                hold_time = (time.time() - pos.get('buy_time', time.time())) / 3600
                                hold_reward = self._calculate_reward(0, hold_time, strategy)

                                import random as _random
                                chosen_hold_action = _random.choice(hold_action_indices)

                                self.model.remember_experience(
                                    state=full_current,
                                    action=chosen_hold_action,
                                    reward=hold_reward,
                                    next_state=full_next,
                                    done=False,
                                    pnl_rub=0
                                )

                    # Блок 2: HOLD для тикеров БЕЗ позиций
                    hold_bonus = hold_config.get('max_bonus', 0.5)
                    no_position_tickers = [
                        t for t in self.current_tickers
                        if t not in self.portfolio.positions
                           and t in self.ticker_states
                           and t in prices
                    ]

                    import random
                    sample_size = min(len(no_position_tickers), max_positions)
                    if sample_size > 0:
                        sampled_tickers = random.sample(no_position_tickers, sample_size)
                        for ticker in sampled_tickers:
                            try:
                                current_price = prices[ticker]
                                current_state = self.ticker_states[ticker]
                                next_state = self._create_next_state(ticker, current_price)

                                fallback_strategy = self.rl_config.get('fallback_strategy', 'balanced')
                                strategy_params = self.model.strategies.get(
                                    fallback_strategy,
                                    self.model.strategies.get('balanced', {})
                                )

                                full_current = self.model._create_strategy_state(current_state, strategy_params)
                                full_next = self.model._create_strategy_state(next_state, strategy_params)

                                import random as _random2
                                chosen_hold_action = _random2.choice(hold_action_indices)

                                self.model.remember_experience(
                                    state=full_current,
                                    action=chosen_hold_action,
                                    reward=hold_bonus,
                                    next_state=full_next,
                                    done=False,
                                    pnl_rub=0
                                )
                            except Exception as e:
                                logger.debug(f"Не удалось записать HOLD-опыт для {ticker}: {e}")

            if filtered_signals and self.risk_manager.check_daily_limits():
                self._execute_trading_decisions(filtered_signals, prices, securities)

            if self.cycle_count % 5 == 0:
                self._rebalance_portfolio(prices, securities)

            if self.cycle_count % 10 == 0:
                self._periodic_learning()

            if self.cycle_count % self.strategy_adaptation_cycles == 0:
                self._adapt_strategies_for_profit()

            if self.cycle_count % 20 == 0:
                self._save_portfolio_state()

            cycle_time = time.time() - cycle_start
            total_value = self.portfolio.get_total_value(prices)

            logger.info(f"Цикл #{self.cycle_count} завершен за {cycle_time:.1f}с | "
                        f"Портфель: {total_value:,.0f}₽ | "
                        f"Позиций: {len(self.portfolio.positions)} | "
                        f"Сигналов: {len(filtered_signals)}")

            # Периодическая сводка PnL
            pnl_summary_interval = self.rl_config.get('logging', {}).get('pnl_summary_interval', 10)
            if self.cycle_count % pnl_summary_interval == 0:
                initial_capital = self.settings.get('initial_capital_rub', 10000)
                total_pnl = total_value - initial_capital
                pnl_percent = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
                daily_trades = getattr(self.risk_manager, 'daily_trades', 0)
                logger.info(f"📈 СВОДКА PNL: {total_pnl:+,.0f}₽ ({pnl_percent:+.2f}%) | Сделок за день: {daily_trades}")

        except Exception as e:
            logger.error(f"Критическая ошибка в run_cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _get_recent_critical_trades(self):
        """Получение критически важных сделок"""
        critical_trades = []

        for ticker, pos in self.portfolio.positions.items():
            current_price = self.moex.get_price(ticker)
            if not current_price:
                continue

            entry_price = pos.get('avg_price', 0)
            if entry_price <= 0:
                continue

            pnl_pct = (current_price - entry_price) / entry_price * 100

            is_critical = (
                    abs(pnl_pct) > 5 or
                    pos.get('strategy') in ['news_aggressive', 'momentum'] or
                    self._has_high_impact_news(ticker)
            )

            if is_critical:
                critical_trades.append({
                    'ticker': ticker,
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'strategy': pos.get('strategy'),
                    'time_held': time.time() - pos.get('buy_time', time.time())
                })

        return critical_trades

    def _get_current_prices(self) -> Dict[str, float]:
        """Получение текущих цен для портфеля"""
        prices = {}

        for ticker in list(self.portfolio.positions.keys()):
            price = self.moex.get_price(ticker)
            if price:
                prices[ticker] = price

        return prices

    def _aggregate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Агрегация и фильтрация сигналов"""
        if not signals:
            return []

        ticker_signals = defaultdict(list)
        for signal in signals:
            ticker = signal['ticker']
            ticker_signals[ticker].append(signal)

        aggregated = []

        for ticker, sig_list in ticker_signals.items():
            action_scores = defaultdict(float)

            for sig in sig_list:
                action = sig['action']
                confidence = sig.get('confidence', 0.5)

                weight = confidence
                if sig.get('reason') == 'news_analysis':
                    weight *= 1.2

                action_scores[action] += weight

            if action_scores:
                best_action = max(action_scores.items(), key=lambda x: x[1])

                if best_action[1] > self.settings.get('model_confidence_threshold', 0.65):
                    aggregated_signal = {
                        'ticker': ticker,
                        'action': best_action[0],
                        'confidence': best_action[1] / len(sig_list),
                        'source_count': len(sig_list),
                        'sources': [s.get('reason', 'unknown') for s in sig_list],
                        'timestamp': datetime.now().isoformat()
                    }
                    aggregated.append(aggregated_signal)

        aggregated.sort(key=lambda x: x['confidence'], reverse=True)

        return aggregated

    def _filter_signals_by_success_rate(self, signals: List[Dict]) -> List[Dict]:
        """Фильтрация сигналов по исторической успешности тикера"""
        filter_config = self.rl_config.get('signal_filter', {})

        if not filter_config.get('enabled', True):
            return signals

        filtered = []
        min_trades = filter_config.get('min_trades_for_filter', 3)
        min_success_rate = filter_config.get('min_success_rate', 0.5)
        new_ticker_confidence_mult = filter_config.get('new_ticker_confidence_mult', 0.7)

        for signal in signals:
            ticker = signal['ticker']
            stats = self.model.ticker_stats.get(ticker, {})
            total_trades = stats.get('total_trades', 0)

            if total_trades >= min_trades:
                if stats.get('success_rate', 0) >= min_success_rate:
                    filtered.append(signal)
            else:
                # Новые тикеры - с пониженной уверенностью
                signal['confidence'] = signal.get('confidence', 0.5) * new_ticker_confidence_mult
                filtered.append(signal)

        logger.debug(f"Отфильтровано сигналов: {len(signals)} -> {len(filtered)}")
        return filtered

    def _record_trade_for_learning(self, ticker: str, action: str, price: float,
                                   quantity: int, confidence: float, pnl: float = 0.0):
        """Запись сделки для обучения модели"""
        try:
            sentiment = self._get_ticker_sentiment(ticker)

            if action == 'BUY':
                self.model.record_trade_outcome(
                    ticker=ticker,
                    action=action,
                    entry_price=price,
                    exit_price=price,
                    hold_time=0.0,
                    news_sentiment=sentiment,
                    market_conditions={
                        'reason': 'signal_execution',
                        'confidence': confidence,
                        'quantity': quantity
                    }
                )
            elif action == 'SELL':
                pos = self.portfolio.positions.get(ticker, {})
                entry_price = pos.get('avg_price', price)
                hold_time = (time.time() - pos.get('buy_time', time.time())) / 3600

                self.model.record_trade_outcome(
                    ticker=ticker,
                    action=action,
                    entry_price=entry_price,
                    exit_price=price,
                    hold_time=hold_time,
                    news_sentiment=sentiment,
                    market_conditions={
                        'reason': 'signal_execution',
                        'confidence': confidence,
                        'pnl': pnl,
                        'quantity': quantity
                    }
                )

        except Exception as e:
            logger.error(f"Ошибка записи сделки для обучения: {e}")

    def check_stops_and_tp(self, prices: Dict[str, float]):
        """Проверка стоп-лоссов и тейк-профитов с трейлинг-стопом"""
        cfg = self.settings

        # Параметры трейлинг-стопа из конфига
        trailing_stop_enabled = cfg.get('trailing_stop_enabled', True)
        trailing_stop_activation = cfg.get('trailing_stop_activation_percent', 3.0)
        trailing_stop_distance = cfg.get('trailing_stop_distance_percent', 2.0)
        partial_take_enabled = cfg.get('partial_take_enabled', True)
        partial_take_percent = cfg.get('partial_take_percent', 3.0)
        partial_take_ratio = cfg.get('partial_take_ratio', 0.5)
        stop_loss_percent = cfg.get('stop_loss_percent', 6.0)
        take_profit_percent = cfg.get('take_profit_percent', 12.0)

        for ticker, pos in list(self.portfolio.positions.items()):
            if ticker not in prices:
                continue

            price = prices[ticker]
            entry_price = pos.get('avg_price', 0.0)

            # Приводим к float (могут быть строки из JSON)
            try:
                entry_price = float(entry_price)
            except (ValueError, TypeError):
                entry_price = 0.0

            if entry_price <= 0:
                continue

            change_pct = (price - entry_price) / entry_price * 100
            hold_time = time.time() - pos.get('buy_time', time.time())

            lot_size = pos.get('lot_size', 1)
            min_step = pos.get('min_step', 0.01)

            if min_step > 0:
                price = round(price / min_step) * min_step

            # ===== ТРЕЙЛИНГ-СТОП =====
            if trailing_stop_enabled and change_pct >= trailing_stop_activation:
                current_stop_raw = pos.get('stop_loss', None)
                if current_stop_raw is None or isinstance(current_stop_raw, str):
                    current_stop = entry_price * (1 - stop_loss_percent / 100.0)
                else:
                    try:
                        current_stop = float(current_stop_raw)
                    except (ValueError, TypeError):
                        current_stop = entry_price * (1 - stop_loss_percent / 100.0)

                trailing_stop_price = price * (1 - trailing_stop_distance / 100.0)

                if trailing_stop_price > current_stop:
                    pos['stop_loss'] = trailing_stop_price
                    pos['trailing_activated'] = True
                    logger.debug(f"Трейлинг-стоп {ticker}: {current_stop:.2f} → {trailing_stop_price:.2f} "
                                 f"(цена: {price:.2f}, +{change_pct:.1f}%)")

            # Проверка обновлённого стоп-лосса
            effective_stop_raw = pos.get('stop_loss', None)
            if effective_stop_raw is None or isinstance(effective_stop_raw, str):
                effective_stop = entry_price * (1 - stop_loss_percent / 100.0)
            else:
                try:
                    effective_stop = float(effective_stop_raw)
                except (ValueError, TypeError):
                    effective_stop = entry_price * (1 - stop_loss_percent / 100.0)

            if price <= effective_stop:
                qty = pos['qty']

                if lot_size > 1 and qty % lot_size != 0:
                    qty = (qty // lot_size) * lot_size
                    if qty == 0:
                        qty = lot_size

                if qty > 0 and self.portfolio.sell(ticker, qty, price):
                    pnl = (price - entry_price) * qty

                    self.risk_manager.update_trade_result(
                        ticker=ticker,
                        action='STOP_LOSS',
                        quantity=qty,
                        price=price,
                        pnl=pnl
                    )

                    logger.warning(f"СТОП-ЛОСС: {ticker} {qty} @ {price:.2f} "
                                   f"({change_pct:+.1f}%, PnL: {pnl:+.0f}₽)"
                                   f"{' [трейлинг]' if pos.get('trailing_activated') else ''}")
                continue

            # ===== ЧАСТИЧНАЯ ФИКСАЦИЯ ПРИБЫЛИ =====
            if partial_take_enabled and change_pct >= partial_take_percent:
                already_partial = pos.get('partial_take_done', False)
                if isinstance(already_partial, str):
                    already_partial = already_partial.lower() == 'true'
                if not already_partial:
                    qty = int(pos['qty'] * partial_take_ratio)

                    if lot_size > 1:
                        qty = (qty // lot_size) * lot_size
                        if qty == 0:
                            qty = lot_size

                    if qty > 0 and qty < pos['qty']:
                        if self.portfolio.sell(ticker, qty, price):
                            pnl = (price - entry_price) * qty

                            self.risk_manager.update_trade_result(
                                ticker=ticker,
                                action='PARTIAL_TAKE',
                                quantity=qty,
                                price=price,
                                pnl=pnl
                            )

                            pos['partial_take_done'] = True
                            pos['stop_loss'] = entry_price * 1.005

                            logger.info(f"ЧАСТИЧНАЯ ФИКСАЦИЯ: {ticker} {qty} @ {price:.2f} "
                                        f"({change_pct:+.1f}%, PnL: {pnl:+.0f}₽, "
                                        f"стоп → {pos['stop_loss']:.2f})")
                continue

            # ===== ТЕЙК-ПРОФИТ (полный) =====
            take_profit_raw = pos.get('take_profit', None)
            if take_profit_raw is None or isinstance(take_profit_raw, str):
                take_profit_price = entry_price * (1 + take_profit_percent / 100.0)
            else:
                try:
                    take_profit_price = float(take_profit_raw)
                except (ValueError, TypeError):
                    take_profit_price = entry_price * (1 + take_profit_percent / 100.0)

            if price >= take_profit_price:
                qty = pos['qty']

                if lot_size > 1:
                    qty = (qty // lot_size) * lot_size
                    if qty == 0:
                        qty = lot_size

                if qty > 0:
                    if self.portfolio.sell(ticker, qty, price):
                        pnl = (price - entry_price) * qty

                        self.risk_manager.update_trade_result(
                            ticker=ticker,
                            action='TAKE_PROFIT',
                            quantity=qty,
                            price=price,
                            pnl=pnl
                        )

                        logger.info(f"ТЕЙК-ПРОФИТ: {ticker} {qty} @ {price:.2f} "
                                    f"({change_pct:+.1f}%, PnL: {pnl:+.0f}₽)")

    def _rebalance_portfolio(self, prices: Dict, securities: Dict):
        """Ребалансировка портфеля с учетом лотности"""
        current_count = len(self.portfolio.positions)
        target_count = self.settings.get('target_positions', 4)

        if current_count < target_count and self.portfolio.cash > self.settings["min_cash_per_trade"]:
            # Получаем сентимент для всех тикеров
            ticker_sentiment = {t: self._get_ticker_sentiment(t) for t in prices.keys()}

            # Строим список кандидатов
            candidates = []
            for ticker, price in prices.items():
                if ticker in self.portfolio.positions:
                    continue

                sentiment = ticker_sentiment.get(ticker, 0)
                volume = securities.get(ticker, {}).get('volume', 0)

                # Простая эвристика: позитивный сентимент + объем
                score = sentiment * 100 + volume / 1e6
                candidates.append((ticker, score))

            candidates.sort(key=lambda x: x[1], reverse=True)

            bought = 0
            for ticker, score in candidates[:5]:
                if bought >= 2:
                    break

                security_info = securities.get(ticker, {})
                lot_size = security_info.get('lot_size', 1)
                min_step = security_info.get('min_step', 0.01)

                price = prices[ticker]

                if min_step > 0:
                    price = round(price / min_step) * min_step

                max_qty_by_cash = int(self.portfolio.cash * 0.15 / price)

                if lot_size > 1:
                    max_qty_by_cash = (max_qty_by_cash // lot_size) * lot_size

                qty = max(lot_size, max_qty_by_cash // 2)

                if lot_size > 1 and qty % lot_size != 0:
                    qty = (qty // lot_size) * lot_size

                if qty >= lot_size:
                    if self.portfolio.buy(ticker, qty, price, 'balanced',
                                          lot_size=lot_size,
                                          min_step=min_step):
                        self.portfolio.positions[ticker]['buy_time'] = time.time()
                        bought += 1

                        logger.info(f"Ребалансировка BUY: {ticker} {qty} @ {price:.2f} "
                                    f"(лот: {lot_size}, score: {score:.2f})")

    def _generate_daily_report(self):
        """Генерация дневного отчета"""
        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'portfolio_value': self.portfolio.get_total_value({}),
                'cash': self.portfolio.cash,
                'positions_count': len(self.portfolio.positions),
                'daily_pnl': self.risk_manager.daily_pnl,
                'daily_trades': self.risk_manager.daily_trades,
                'market_sentiment': self._get_market_sentiment(),
                'signals_generated': len(self.signals_cache),
                'risk_metrics': self.risk_manager.get_risk_metrics()
            }

            report_file = f"data/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Дневной отчет сохранен: {report_file}")

        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")

    def analyze_sentiment(self, force: bool = False):
        """Анализ рыночного настроения"""
        try:
            market_sentiment = self._get_market_sentiment()
            self.model.update_market_sentiment(market_sentiment)
            logger.info(f"Рыночное настроение обновлено: {market_sentiment:+.3f}")
        except Exception as e:
            logger.error(f"Ошибка анализа сентимента: {e}")

    def _create_next_state(self, ticker: str, exit_price: float) -> torch.Tensor:
        try:
            current_price = self.moex.get_price(ticker) or exit_price
            securities = self.moex.get_all_securities()
            security_info = securities.get(ticker, {})
            sentiment = self._get_ticker_sentiment(ticker)

            indicators = {}
            try:
                indicators = self.technical_core.calculate_indicators(ticker)
            except:
                pass

            ticker_names = self.rl_config.get('ticker_names', {})
            keywords = ticker_names.get(ticker, [])
            news_items = self.news_fetcher.search_news(ticker=ticker, limit=3, keywords=keywords)
            news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
            news_features = self.model.encode_news(news_texts)

            market_sentiment = self._get_market_sentiment()
            macro_data = self.moex.get_macro_data()

            settings = self.settings
            default_spread = settings.get('default_spread', 0.01)
            default_rsi = settings.get('default_rsi', 50)
            default_volatility = settings.get('default_volatility', 0.1)
            default_bb_position = settings.get('default_bb_position', 0.5)

            norm = self.model.normalization

            enhanced_market_data = {
                'volume': indicators.get('volume', 0) if indicators else 0,
                'spread': security_info.get('spread', default_spread),
                'rsi': indicators.get('rsi', default_rsi) if indicators else default_rsi,
                'volatility': indicators.get('atr',
                                             0) / current_price if indicators and current_price > 0 else default_volatility,
                'sma_10_ratio': indicators.get('sma_10',
                                               current_price) / current_price if indicators and current_price > 0 else 1.0,
                'sma_20_ratio': indicators.get('sma_20',
                                               current_price) / current_price if indicators and current_price > 0 else 1.0,
                'bb_position': indicators.get('bb_position',
                                              default_bb_position) if indicators else default_bb_position,
                'volume_ratio': indicators.get('volume_ratio', 1.0) if indicators else 1.0,
                'atr': indicators.get('atr', 0) if indicators else 0,
                'market_cap': security_info.get('market_cap', 0),
                'lot_size': security_info.get('lot_size', 1),
                'min_step': security_info.get('min_step', 0.01),
                'sector': security_info.get('sector', 'other'),
                'momentum': security_info.get('momentum', 0.0),
                'imoex': macro_data.get('imoex', 0),
                'imoex_change': macro_data.get('imoex_change', 0),
                'rtsi': macro_data.get('rtsi', 0),
                'rtsi_change': macro_data.get('rtsi_change', 0),
                'rvi': macro_data.get('rvi', 20.0),
                'rvi_change': macro_data.get('rvi_change', 0),
                'moexog': macro_data.get('moexog', 0),
                'moexfn': macro_data.get('moexfn', 0),
                'brent': macro_data.get('brent', 0),
                'brent_change': macro_data.get('brent_change', 0),
                'market_liquidity_ratio': macro_data.get('market_liquidity_ratio', 0.0),
                'market_activity_score': macro_data.get('market_activity_score', 0.0),
                # MARKET_FEATURES — нормализация из конфига
                'spread_pct': (security_info.get('spread', 0) / current_price) if current_price > 0 else 0.0,
                'market_mood': macro_data.get('market_mood', 0.0),
                'shares_turnover': macro_data.get('shares_turnover', 0) / norm.get('shares_turnover_divisor', 1e12),
                'rvi_normalized': macro_data.get('rvi', 20.0) / norm.get('rvi_divisor', 100.0),
                'imoex_normalized': macro_data.get('imoex', 0) / norm.get('imoex_divisor', 4000.0),
                'market_cap_total': macro_data.get('market_cap', 0) / norm.get('market_cap_divisor_total', 1e14),
                'liquidity_ratio': macro_data.get('market_liquidity_ratio', 0.0),
                'cbr_rate_normalized': macro_data.get('cbr_rate', 0.0) / norm.get('cbr_rate_divisor', 20.0),
                'vix': macro_data.get('vix', 0.0) / norm.get('vix_divisor', 50.0),
                'moexog_normalized': macro_data.get('moexog', 0) / norm.get('moexog_divisor', 10000.0),
            }

            state = self.model.build_state_vector(
                ticker=ticker,
                price=current_price,
                momentum=security_info.get('momentum', 0.0),
                sentiment=sentiment,
                news_features=news_features,
                market_data=enhanced_market_data,
                market_sentiment=market_sentiment,
                portfolio=self.portfolio
            )

            return state.to(self.model.device)

        except Exception as e:
            logger.error(f"Ошибка создания следующего состояния для {ticker}: {e}")
            return torch.zeros(self.model.total_state_dim, dtype=torch.float32).to(self.model.device)

    def _adapt_strategies_for_profit(self):
        """Адаптация стратегий - ТОЛЬКО на основе реальных результатов"""
        try:
            for strategy_name, perf in self.model.strategy_performance.items():
                if perf['total_trades'] >= 5:  # минимальная статистика
                    win_rate = perf['win_rate']
                    avg_pnl = perf['avg_pnl']

                    strategy = self.model.strategies[strategy_name]

                    # 📈 УСПЕШНАЯ СТРАТЕГИЯ - усиливаем пропорционально прибыли
                    if avg_pnl > 0:
                        # Сила усиления зависит от размера прибыли
                        strength = min(avg_pnl * 10, 0.5)  # максимум +50%

                        # Увеличиваем риск, но не больше 2.0
                        old_risk = strategy['risk_multiplier']
                        strategy['risk_multiplier'] = min(old_risk * (1 + strength), 2.0)

                        # Уменьшаем время удержания (быстрее фиксируем прибыль)
                        old_hold = strategy['target_hold_time_hours']
                        strategy['target_hold_time_hours'] = max(old_hold * (1 - strength * 0.5), 0.5)

                        logger.info(f"📈 Усиление {strategy_name}: "
                                    f"risk {old_risk:.2f}→{strategy['risk_multiplier']:.2f} "
                                    f"(+{strength:.1%}, avg_pnl={avg_pnl:.2%})")

                    # 📉 УБЫТОЧНАЯ СТРАТЕГИЯ - ослабляем пропорционально убытку
                    elif avg_pnl < 0:
                        # Сила ослабления зависит от размера убытка
                        weakness = min(abs(avg_pnl) * 10, 0.5)  # максимум -50%

                        # Уменьшаем риск, но не меньше 0.5
                        old_risk = strategy['risk_multiplier']
                        strategy['risk_multiplier'] = max(old_risk * (1 - weakness), 0.5)

                        # Увеличиваем время удержания (даём больше времени на разворот)
                        old_hold = strategy['target_hold_time_hours']
                        strategy['target_hold_time_hours'] = min(old_hold * (1 + weakness), 24)

                        logger.info(f"📉 Ослабление {strategy_name}: "
                                    f"risk {old_risk:.2f}→{strategy['risk_multiplier']:.2f} "
                                    f"(-{weakness:.1%}, avg_pnl={avg_pnl:.2%})")

                    # Адаптация стоп-лосса под волатильность (объективный фактор)
                    if hasattr(self.model, 'volatility_index'):
                        old_stop = strategy.get('stop_loss_percent', 2.5)
                        old_take = strategy.get('take_profit_percent', 5.0)

                        # Базовая настройка: 2% на стоп, 4% на тейк при нормальной волатильности
                        base_stop = 2.0
                        base_take = 4.0

                        # Корректировка под текущую волатильность
                        vol_factor = 1.0 + self.model.volatility_index
                        strategy['stop_loss_percent'] = base_stop * vol_factor
                        strategy['take_profit_percent'] = base_take * vol_factor

                        logger.debug(f"📊 Адаптация под волатильность {strategy_name}: "
                                     f"vol={self.model.volatility_index:.2f}, "
                                     f"stop {old_stop:.1f}→{strategy['stop_loss_percent']:.1f}%, "
                                     f"take {old_take:.1f}→{strategy['take_profit_percent']:.1f}%")

            # Сохраняем обновленные стратегии
            self.model.save_model()
            logger.info("✅ Стратегии адаптированы на основе реальных результатов")

        except Exception as e:
            logger.error(f"Ошибка адаптации стратегий: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Получение сводки портфеля"""
        prices = self._get_current_prices()
        total_value = self.portfolio.get_total_value(prices)

        """ Расчет pending комиссий"""
        pending_total = 0
        pending_count = 0
        if hasattr(self.portfolio, 'pending_commissions'):
            for comm in self.portfolio.pending_commissions:
                if not comm.get('processed', False):
                    pending_total += comm['amount']
                    pending_count += 1

        summary = {
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'reserved_cash': self.portfolio.reserved_cash,
            'available_cash': self.portfolio.cash - self.portfolio.reserved_cash,
            'pending_commissions': {
                'count': pending_count,
                'total': pending_total,
                'next_settlement': self._get_next_settlement_date()
            },
            'positions_count': len(self.portfolio.positions),
            'initial_capital': self.settings["initial_capital_rub"],
            'total_pnl': total_value - self.settings["initial_capital_rub"],
            'pnl_percent': ((total_value / self.settings["initial_capital_rub"]) - 1) * 100,
            'current_signals': self.signals_cache[:5],
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'session_info': self.scheduler.get_session_info(),
            'last_update': datetime.now().isoformat(),
            'market_sentiment': self._get_market_sentiment(),
            'news_stats': self.news_fetcher.stats if hasattr(self.news_fetcher, 'stats') else {}
        }

        # Детали по позициям
        positions_detail = []
        for ticker, pos in self.portfolio.positions.items():
            current_price = prices.get(ticker, pos['avg_price'])
            position_value = pos['qty'] * current_price

            positions_detail.append({
                'ticker': ticker,
                'quantity': pos['qty'],
                'avg_price': pos['avg_price'],
                'current_price': current_price,
                'position_value': position_value,
                'pnl': (current_price - pos['avg_price']) * pos['qty'],
                'pnl_percent': ((current_price / pos['avg_price']) - 1) * 100,
                'weight': (position_value / total_value * 100) if total_value > 0 else 0,
                'strategy': pos.get('strategy', 'unknown'),  # ← ДОБАВИТЬ
                'buy_time': pos.get('buy_time')  # ← ДОБАВИТЬ (для дней удержания)
            })

        summary['positions'] = positions_detail

        return summary

    def _get_next_settlement_date(self) -> Optional[str]:
        """Получение ближайшей даты списания"""
        if not hasattr(self.portfolio, 'pending_commissions'):
            return None

        upcoming = [
            c for c in self.portfolio.pending_commissions
            if not c.get('processed', False)
        ]

        if not upcoming:
            return None

        upcoming.sort(key=lambda x: x['settlement_date'])
        return upcoming[0]['settlement_date']

    def update_settings(self, new_settings: Dict):
        """Обновление настроек системы"""
        try:
            self.settings.update(new_settings)

            updates_to_apply = []

            if 'initial_capital_rub' in new_settings:
                self.portfolio.initial_capital = new_settings['initial_capital_rub']
                updates_to_apply.append(f"initial_capital_rub={new_settings['initial_capital_rub']}")

            if 'max_positions' in new_settings:
                self.portfolio.max_positions = new_settings['max_positions']
                updates_to_apply.append(f"max_positions={new_settings['max_positions']}")

            if 'risk_per_trade_percent' in new_settings:
                if hasattr(self.risk_manager, 'config'):
                    self.risk_manager.config['risk_per_trade_percent'] = new_settings['risk_per_trade_percent']
                    updates_to_apply.append(f"risk_per_trade_percent={new_settings['risk_per_trade_percent']}%")

            stop_loss_params = ['stop_loss_percent', 'take_profit_percent', 'max_position_weight_percent']
            for param in stop_loss_params:
                if param in new_settings:
                    if hasattr(self.risk_manager, 'config') and param in self.risk_manager.config:
                        self.risk_manager.config[param] = new_settings[param]
                        updates_to_apply.append(f"{param}={new_settings[param]}")

            if updates_to_apply:
                logger.info(f"Настройки обновлены: {', '.join(updates_to_apply)}")
            else:
                logger.info(f"Настройки обновлены: {new_settings}")

        except Exception as e:
            logger.error(f"Ошибка обновления настроек брокера: {e}")

    def reload_configs(self):
        """Перезагрузка конфигов на лету без перезапуска"""
        try:
            # Перезагружаем settings.json
            with open('config/settings.json', 'r', encoding='utf-8') as f:
                new_settings = json.load(f)

            # Обновляем настройки брокера
            self.settings.update(new_settings)

            # Обновляем Risk Manager
            if hasattr(self, 'risk_manager'):
                self.risk_manager.config = self.risk_manager._load_config("config/settings.json")
                self.risk_manager.config.update({
                    'stop_loss_percent': new_settings.get('stop_loss_percent', 6.0),
                    'take_profit_percent': new_settings.get('take_profit_percent', 12.0),
                    'min_cash_per_trade': new_settings.get('min_cash_per_trade', 1000),
                    'max_daily_trades': new_settings.get('max_daily_trades', 20),
                    'max_positions': new_settings.get('max_positions', 10),
                    'max_position_weight_percent': new_settings.get('max_position_weight_percent', 30),
                    'daily_loss_limit_percent': new_settings.get('daily_loss_limit_percent', 5),
                    'risk_per_trade_percent': new_settings.get('risk_per_trade_percent', 3.0),
                })

            # Обновляем Portfolio
            if hasattr(self, 'portfolio'):
                self.portfolio.max_positions = new_settings.get('max_positions', 10)
                self.portfolio.max_trades_per_hour = new_settings.get('max_trades_per_hour', 10)
                self.portfolio.daily_commission_limit = new_settings.get('daily_commission_limit', 100.0)

            # Перезагружаем strategies.json
            with open('config/strategies.json', 'r', encoding='utf-8') as f:
                new_strategies = json.load(f)

            if hasattr(self, 'model') and self.model:
                # Обновляем стратегии
                if 'strategies' in new_strategies:
                    for name, params in new_strategies['strategies'].items():
                        if name in self.model.strategies:
                            # Сохраняем обученный risk_multiplier, если он ниже конфигового
                            old_risk = self.model.strategies[name].get('risk_multiplier', 1.0)
                            new_risk = params.get('risk_multiplier', 1.0)
                            self.model.strategies[name].update(params)
                            if old_risk < new_risk:
                                self.model.strategies[name]['risk_multiplier'] = old_risk

                # Обновляем confidence_boost_factor
                strategy_selection = new_strategies.get('strategy_selection', {})
                if 'confidence_boost_factor' in strategy_selection:
                    self.model.confidence_boost_factor = strategy_selection['confidence_boost_factor']

            # Перезагружаем rl_config.json
            if hasattr(self, 'model') and self.model:
                with open('config/rl_config.json', 'r', encoding='utf-8') as f:
                    new_rl = json.load(f)

                # Обновляем exploration
                if 'exploration' in new_rl:
                    self.model.exploration_rate = new_rl['exploration'].get('initial_exploration_rate', 0.03)

                # Обновляем hold_reward
                if 'hold_reward' in new_rl:
                    self.rl_config['hold_reward'] = new_rl['hold_reward']

                # Обновляем reward_config
                if 'reward_config' in new_rl:
                    self.rl_config['reward_config'] = new_rl['reward_config']

                # Обновляем signal_filter
                if 'signal_filter' in new_rl:
                    self.rl_config['signal_filter'] = new_rl['signal_filter']

            logger.info("✅ Конфиги перезагружены на лету")

        except Exception as e:
            logger.error(f"❌ Ошибка перезагрузки конфигов: {e}")


    # ============================================
    # БЭКОФИС ФУНКЦИИ
    # ============================================

    def check_clearing_liquidity(self):
        """16:00 - Проверка ликвидности"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: ПРОВЕРКА ЛИКВИДНОСТИ (16:00)")
        logger.info("=" * 60)

        try:
            prices = self._get_current_prices()
            total_exposure = 0
            positions_detail = []

            for ticker, pos in self.portfolio.positions.items():
                current_price = prices.get(ticker, pos['avg_price'])
                position_value = pos['qty'] * current_price
                total_exposure += position_value
                positions_detail.append({
                    'ticker': ticker,
                    'value': position_value,
                    'qty': pos['qty'],
                    'price': current_price
                })

            liquidity_ratio = self.portfolio.cash / total_exposure if total_exposure > 0 else 1.0
            min_ratio = self.settings.get('back_office', {}).get('min_liquidity_ratio', 0.2)

            report = {
                'timestamp': datetime.now().isoformat(),
                'cash': self.portfolio.cash,
                'total_exposure': total_exposure,
                'liquidity_ratio': liquidity_ratio,
                'min_required_ratio': min_ratio,
                'is_sufficient': liquidity_ratio >= min_ratio,
                'positions_count': len(self.portfolio.positions),
                'positions': positions_detail
            }

            logger.info(f"💰 Кэш: {self.portfolio.cash:,.0f}₽")
            logger.info(f"📊 Обязательства: {total_exposure:,.0f}₽")
            logger.info(f"⚖ Коэффициент ликвидности: {liquidity_ratio:.2%} (мин: {min_ratio:.2%})")

            if liquidity_ratio < min_ratio:
                logger.warning(f"⚠ КРИТИЧЕСКИ НИЗКАЯ ЛИКВИДНОСТЬ!")
                self._send_liquidity_alert(report)
                # ✅ НЕ ОСТАНАВЛИВАЕМ ТОРГОВЛЮ! ТОЛЬКО ПРЕДУПРЕЖДАЕМ
            else:
                logger.info(f"✅ Ликвидность достаточная")

            self._save_liquidity_report(report)

            # ✅ ПРИНУДИТЕЛЬНО ВКЛЮЧАЕМ ТОРГОВЛЮ
            self.trading_enabled = True

            return report

        except Exception as e:
            logger.error(f"❌ Ошибка проверки ликвидности: {e}")
            self.trading_enabled = True  # При ошибке тоже включаем
            return {'error': str(e)}

    def execute_z0_deadline(self):
        """16:50 - Дедлайн сделок Z0"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: ДЕДЛАЙН Z0 (16:50)")
        logger.info("=" * 60)

        try:
            back_config = self.settings.get('back_office', {})
            auto_close = back_config.get('auto_close_positions_at_deadline', False)

            if not auto_close:
                logger.info("⏸ Автоматическое закрытие позиций отключено")
                return {'action': 'skipped', 'reason': 'auto_close_disabled'}

            liquidity = self.check_clearing_liquidity()

            if liquidity.get('is_sufficient', True):
                logger.info("✅ Ликвидность достаточная, закрытие не требуется")
                return {'action': 'none', 'liquidity': liquidity}

            max_close_percent = back_config.get('max_position_close_percent', 50)
            closed_positions = []

            prices = self._get_current_prices()

            for ticker, pos in list(self.portfolio.positions.items()):
                if ticker not in prices:
                    continue

                qty_to_sell = int(pos['qty'] * max_close_percent / 100)
                lot_size = pos.get('lot_size', 1)

                if lot_size > 1:
                    qty_to_sell = (qty_to_sell // lot_size) * lot_size

                if qty_to_sell >= lot_size:
                    if self.portfolio.sell(ticker, qty_to_sell, prices[ticker]):
                        closed_positions.append({
                            'ticker': ticker,
                            'qty': qty_to_sell,
                            'price': prices[ticker]
                        })
                        logger.info(f"📉 Частично закрыта {ticker}: {qty_to_sell} @ {prices[ticker]:.2f}")

            result = {
                'action': 'partial_close',
                'closed_positions': closed_positions,
                'cash_after': self.portfolio.cash
            }

            logger.info(f"✅ Дедлайн Z0 выполнен, закрыто {len(closed_positions)} позиций")
            return result

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения дедлайна Z0: {e}")
            return {'error': str(e)}

    def process_clearing_17(self):
        """17:00 - Первый клиринг"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: КЛИРИНГ 17:00")
        logger.info("=" * 60)

        try:
            prices = self._get_current_prices()
            portfolio_value = self.portfolio.get_total_value(prices)

            state = {
                'timestamp': datetime.now().isoformat(),
                'clearing': '17:00',
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions': len(self.portfolio.positions),
                'open_pnl': portfolio_value - self.portfolio.cash - sum(
                    p['qty'] * p['avg_price'] for p in self.portfolio.positions.values()
                )
            }

            self._save_clearing_state(state)

            logger.info(f"📊 Портфель на 17:00: {portfolio_value:,.0f}₽")
            logger.info(f"💰 Кэш: {self.portfolio.cash:,.0f}₽")

            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: принудительно включаем торговлю
            self.trading_enabled = True
            logger.info("✅ Торговля принудительно включена после клиринга 17:00")

            # ✅ Принудительный сброс буфера логов
            for handler in logger.logger.handlers:
                try:
                    handler.flush()
                except:
                    pass

            return state

        except Exception as e:
            logger.error(f"❌ Ошибка клиринга 17:00: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # ✅ ДАЖЕ ПРИ ОШИБКЕ ВКЛЮЧАЕМ ТОРГОВЛЮ
            self.trading_enabled = True
            logger.info("✅ Торговля принудительно включена после ошибки клиринга")

            return {'error': str(e), 'action': 'forced_enabled'}

    def process_clearing_19(self):
        """19:00 - Расчет комиссии и постановка в очередь"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: КЛИРИНГ 19:00")
        logger.info("=" * 60)

        try:
            daily_trades = getattr(self.portfolio, 'daily_trades', [])
            total_turnover = sum(t.get('value', 0) for t in daily_trades)
            total_reserved = getattr(self.portfolio, 'commission_spent_today', 0.0)

            if total_reserved == 0:
                logger.info("💰 Комиссия не начислена (нет сделок)")
                # ✅ ВКЛЮЧАЕМ ТОРГОВЛЮ ДАЖЕ ЕСЛИ НЕТ КОМИССИЙ
                self.trading_enabled = True
                return {'commission': 0, 'turnover': 0}

            today = datetime.now()
            weekday_map = self.tbank_config.get('weekday_settlement_map', {"4": 3, "5": 2, "6": 1})
            days_to_add = self.tbank_config.get('default_settlement_days', 1)
            weekday_str = str(today.weekday())
            if weekday_str in weekday_map:
                days_to_add = weekday_map[weekday_str]

            settlement_date = today + timedelta(days=days_to_add)

            commission_record = {
                'date': today.strftime('%Y-%m-%d'),
                'amount': total_reserved,
                'turnover': total_turnover,
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'settlement_time': self.tbank_settlement_time,
                'processed': False,
                'created_at': datetime.now().isoformat()
            }

            if not hasattr(self.portfolio, 'pending_commissions'):
                self.portfolio.pending_commissions = []

            self.portfolio.pending_commissions.append(commission_record)

            logger.info(f"📅 Комиссия за {today.strftime('%d.%m.%Y')}: {total_reserved:,.2f}₽")
            logger.info(f"📅 Резерв заморожен: {self.portfolio.reserved_cash:,.2f}₽")
            logger.info(f"📅 Будет списана {settlement_date.strftime('%d.%m.%Y')} до {self.tbank_settlement_time}")

            if hasattr(self.portfolio, 'reset_daily_trades'):
                self.portfolio.reset_daily_trades()

            self._generate_daily_report()

            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: принудительно включаем торговлю
            self.trading_enabled = True
            logger.info("✅ Торговля принудительно включена после клиринга 19:00")

            return {
                'commission': total_reserved,
                'turnover': total_turnover,
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'settlement_time': self.tbank_settlement_time,
                'reserved_cash': self.portfolio.reserved_cash
            }

        except Exception as e:
            logger.error(f"❌ Ошибка клиринга 19:00: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # ✅ ДАЖЕ ПРИ ОШИБКЕ ВКЛЮЧАЕМ ТОРГОВЛЮ
            self.trading_enabled = True
            return {'error': str(e), 'action': 'forced_enabled'}

    def process_pending_commissions(self):
        """Проверка и списание pending комиссий"""
        if not hasattr(self.portfolio, 'pending_commissions'):
            return

        today = datetime.now().date()
        processed_count = 0
        total_commission = 0

        logger.debug(f"🔍 Проверка pending комиссий: {len(self.portfolio.pending_commissions)} записей")

        for comm in list(self.portfolio.pending_commissions):
            if comm.get('processed', False):
                continue

            settlement_date = datetime.strptime(comm['settlement_date'], '%Y-%m-%d').date()

            if settlement_date <= today:
                logger.info(
                    f"💸 Списание комиссии {comm['amount']:,.2f}₽ за {comm['date']} (дата списания: {comm['settlement_date']})")

                if self.portfolio.reserved_cash >= comm['amount']:
                    self.portfolio.cash -= comm['amount']
                    self.portfolio.reserved_cash -= comm['amount']
                    comm['processed'] = True
                    comm['processed_date'] = today.strftime('%Y-%m-%d')
                    comm['processed_time'] = datetime.now().strftime('%H:%M')

                    processed_count += 1
                    total_commission += comm['amount']

                    logger.info(f"💸 СПИСАНО: комиссия {comm['amount']:,.2f}₽ за {comm['date']}")
                    logger.info(f"💰 Кэш после списания: {self.portfolio.cash:,.2f}₽")
                    logger.info(f"💰 Резерв после списания: {self.portfolio.reserved_cash:,.2f}₽")
                else:
                    logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: недостаточно резерва для комиссии {comm['amount']:,.2f}₽!")
                    logger.error(f"💰 Резерв: {self.portfolio.reserved_cash:,.2f}₽")

                    # Принудительное списание
                    self.portfolio.cash -= (comm['amount'] - self.portfolio.reserved_cash)
                    self.portfolio.reserved_cash = 0
                    comm['processed'] = True
                    comm['forced'] = True
                    comm['forced_date'] = today.strftime('%Y-%m-%d')

                    processed_count += 1
                    total_commission += comm['amount']
                    logger.warning(f"⚠️ ПРИНУДИТЕЛЬНО списана комиссия {comm['amount']:,.2f}₽")

        if processed_count > 0:
            logger.info(f"✅ Обработано {processed_count} комиссий на сумму {total_commission:,.2f}₽")
            logger.info(f"💰 Остаток резерва: {self.portfolio.reserved_cash:,.2f}₽")

        # Очищаем обработанные комиссии
        self.portfolio.pending_commissions = [
            c for c in self.portfolio.pending_commissions
            if not c.get('processed', False)
        ]

    def _send_liquidity_alert(self, report: Dict):
        """Отправка alert о низкой ликвидности"""
        try:
            back_config = self.settings.get('back_office', {})
            if not back_config.get('alert_on_low_liquidity', True):
                return

            email = back_config.get('alert_email', 'backoffice@example.com')

            logger.warning(f"📧 ALERT: Низкая ликвидность! Требуется довнесение средств")
            logger.warning(f"📧 Кому: {email}")
            logger.warning(f"📧 Детали: cash={report['cash']:.0f}₽, exposure={report['total_exposure']:.0f}₽")

        except Exception as e:
            logger.error(f"Ошибка отправки alert: {e}")

    def _save_liquidity_report(self, report: Dict):
        """Сохранение отчета о ликвидности"""
        try:
            import os
            os.makedirs("data/backoffice", exist_ok=True)

            filename = f"data/backoffice/liquidity_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"💾 Отчет ликвидности сохранен: {filename}")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")

    def _save_clearing_state(self, state: Dict):
        """Сохранение состояния клиринга"""
        try:
            import os
            os.makedirs("data/backoffice", exist_ok=True)

            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"data/backoffice/clearing_{date_str}.json"

            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

            history.append(state)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)

            logger.info(f"💾 Состояние клиринга сохранено")

        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _fixate_daily_profit(self):
        """Фиксация дневной прибыли"""
        profit_config = self.settings.get('daily_profit_fixation', {})
        if not profit_config.get('enabled', False):
            return

        try:
            total_value = self.portfolio.get_total_value({})
            daily_start = getattr(self, 'daily_start_capital', None)

            if daily_start is None or daily_start <= 0:
                logger.debug("Нет daily_start_capital, фиксация невозможна")
                return

            daily_profit = total_value - daily_start
            min_profit = profit_config.get('min_profit_to_fix', 50.0)
            reinvest_pct = profit_config.get('reinvest_percent', 0.0)

            if daily_profit >= min_profit:
                # Фиксируем прибыль
                fixated = daily_profit * (1.0 - reinvest_pct)
                reinvested = daily_profit * reinvest_pct

                self.portfolio.reserved_cash += fixated
                self.portfolio.cash -= fixated
                self.daily_start_capital = daily_start + reinvested

                # Сохраняем в историю
                if not hasattr(self.portfolio, 'daily_profit_history'):
                    self.portfolio.daily_profit_history = []

                self.portfolio.daily_profit_history.append({
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'profit': daily_profit,
                    'fixated': fixated,
                    'reinvested': reinvested,
                    'total_value': total_value,
                    'timestamp': datetime.now().isoformat()
                })

                logger.info(f"💰 ДНЕВНАЯ ПРИБЫЛЬ ЗАФИКСИРОВАНА: "
                            f"+{daily_profit:+.0f}₽ "
                            f"(выведено: {fixated:.0f}₽, "
                            f"реинвестировано: {reinvested:.0f}₽)")
            else:
                logger.info(f"Прибыль {daily_profit:+.0f}₽ меньше порога {min_profit:.0f}₽ — переносим")

        except Exception as e:
            logger.error(f"Ошибка фиксации прибыли: {e}")

    def _set_daily_start_capital(self):
        """Установка начального капитала дня"""
        total_value = self.portfolio.get_total_value({})
        self.daily_start_capital = total_value
        logger.info(f"Начальный капитал дня: {total_value:,.0f}₽")


    def get_sentiment_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории сентимента для веб-дашборда"""
        try:
            # Получаем новости с сентиментом
            news_items = self.news_fetcher.get_last_news(limit=limit)

            if not news_items:
                return []

            # Анализируем сентимент
            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            # Преобразуем в формат для дашборда
            sentiment_history = []
            for news in news_with_sentiment:
                # Извлекаем тикеры из новости
                ticker = self._extract_ticker_from_news(news)

                sentiment_history.append({
                    'timestamp': news.get('published', datetime.now().isoformat()),
                    'sentiment': news.get('sentiment', 0.0),
                    'source': news.get('source', 'unknown'),
                    'ticker': ticker or 'MARKET',
                    'title': news.get('title', '')[:50]  # Для подсказки
                })

            # Сортируем по времени (новые сначала)
            sentiment_history.sort(key=lambda x: x['timestamp'], reverse=True)

            return sentiment_history[:limit]

        except Exception as e:
            logger.error(f"Ошибка получения истории сентимента: {e}")
            return []

    def _extract_ticker_from_news(self, news: Dict) -> Optional[str]:
        """Извлечение тикера из новости с использованием всех доступных тикеров"""
        try:
            title = news.get('title', '').upper()

            # 🔥 ПОЛУЧАЕМ ВСЕ ТИКЕРЫ ИЗ ПОРТФЕЛЯ И ТЕКУЩЕГО СПИСКА
            all_tickers = set()

            # Добавляем тикеры из портфеля
            if hasattr(self, 'portfolio') and self.portfolio.positions:
                all_tickers.update(self.portfolio.positions.keys())

            # Добавляем тикеры из текущего списка
            if hasattr(self, 'current_tickers') and self.current_tickers:
                all_tickers.update(self.current_tickers)

            # Добавляем популярные тикеры как fallback
            popular_tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR',
                               'GMKN', 'NVTK', 'YNDX', 'TATN', 'PLZL',
                               'MTSS', 'MOEX', 'AFLT', 'MGNT', 'NLMK']
            all_tickers.update(popular_tickers)

            # Ищем любой тикер в заголовке
            for ticker in all_tickers:
                # Ищем как отдельное слово (с границами слова)
                if f" {ticker} " in f" {title} ":
                    return ticker
                # Ищем в начале строки
                if title.startswith(ticker + " "):
                    return ticker
                # Ищем в конце строки
                if title.endswith(" " + ticker):
                    return ticker
                # Простое вхождение (для коротких заголовков)
                if ticker in title and len(ticker) > 3:  # Избегаем ложных срабатываний
                    return ticker

            # Проверяем расшифровки компаний
            company_names = {
                'СБЕР': 'SBER',
                'ГАЗПРОМ': 'GAZP',
                'ЛУКОЙЛ': 'LKOH',
                'РОСНЕФТЬ': 'ROSN',
                'ВТБ': 'VTBR',
                'НОРНИКЕЛЬ': 'GMKN',
                'НОВАТЭК': 'NVTK',
                'ЯНДЕКС': 'YNDX',
                'ТАТНЕФТЬ': 'TATN',
                'ПОЛЮС': 'PLZL'
            }

            for name, ticker in company_names.items():
                if name in title:
                    return ticker

            return None

        except Exception as e:
            logger.error(f"Ошибка извлечения тикера: {e}")
            return None

    def shutdown(self):
        """Корректное завершение работы с ПОЛНЫМ СОХРАНЕНИЕМ ОПЫТА"""
        logger.info("=" * 60)
        logger.info("🛑 ЗАВЕРШЕНИЕ РАБОТЫ SMART BROKER")
        logger.info("=" * 60)

        try:
            # 1. Сохраняем все незавершенные опыты
            pending_count = len(self.pending_experiences)
            if pending_count > 0:
                logger.info(f"Завершаем {pending_count} незавершенных опытов...")
                for exp in list(self.pending_experiences):
                    if not exp.get('completed', False):
                        current_price = self.moex.get_price(exp['ticker'])
                        if current_price:
                            self._complete_rl_experience(exp['ticker'], current_price)
                        else:
                            # Если не можем получить цену, завершаем принудительно
                            exp['completed'] = True
                            logger.warning(f"Принудительно завершен опыт {exp['ticker']}")

            # 2. Сохраняем приоритетную память
            if hasattr(self.model, 'prioritized_buffer') and self.model.prioritized_buffer.size > 0:
                logger.info(f"Сохраняем приоритетную память: {self.model.prioritized_buffer.size} опытов")
                # Приоритетная память сохраняется через обычный save_memory

            # 3. Сохраняем обычную память
            if len(self.model.memory) > 0:
                self.model.save_memory()
                logger.info(f"✅ Память модели сохранена: {len(self.model.memory)} опытов")

            # 4. Сохраняем модель
            self.model.save_model()
            logger.info("✅ Модель сохранена")

            # 5. Сохраняем портфель
            self._save_portfolio_state()
            logger.info("✅ Портфель сохранен")

            # 6. Останавливаем фоновые процессы
            if hasattr(self, 'trainer') and self.trainer:
                self.trainer.save_memory()
                logger.info("✅ Память тренера сохранена")

            logger.info("=" * 60)
            logger.info("✅ SmartBroker успешно завершил работу")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Ошибка при завершении: {e}")
            import traceback
            logger.error(traceback.format_exc())