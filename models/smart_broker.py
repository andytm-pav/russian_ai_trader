"""
Главный модуль Smart Broker с интеграцией всех компонентов
"""

import json
import torch
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from core.core_news_trader import NewsTraderCore
from core.core_technical_trader import TechnicalTraderCore
from core.risk_manager import RiskManager
from core.trading_hours_scheduler import TradingScheduler
from fetchers.moex_fetcher import MoexFetcher
from fetchers.rss_fetcher import RSSFetcher
from utils.portfolio_manager import PortfolioManager
from utils.logger import logger
from models.trader_model import trader_model_instance


class SmartPortfolioBroker:
    """Умный брокер с интеграцией всех торговых ядер"""

    def __init__(self, settings: Dict):
        self.settings = settings
        self.moex = MoexFetcher()
        self.rss_fetcher = RSSFetcher()
        self.news_core = NewsTraderCore()
        self.technical_core = TechnicalTraderCore()
        self.risk_manager = RiskManager()
        self.scheduler = TradingScheduler()
        self.portfolio = PortfolioManager()
        self.model = trader_model_instance

        # Инициализация с начальным капиталом
        self.portfolio.cash = settings.get("initial_capital_rub", 10000)

        # Торговые переменные
        self.last_cycle_time = 0
        self.cycle_count = 0
        self.trading_enabled = True
        self.signals_cache = []
        self.current_tickers = []

        # Запуск компонентов
        self._initialize_components()

        print(f"[SmartBroker] Инициализирован. Капитал: {self.portfolio.cash:,.0f}₽")
        print(f"[SmartBroker] Модель sentiment: {self.model.market_sentiment:.3f}")
        print(f"[SmartBroker] Макс. позиций: {settings['max_positions']}")

        self.ticker_states = {}  # {ticker: last_state}
        self.pending_experiences = []  # Опыты для обучения
        self.strategy_tracker = defaultdict(list)  # История стратегий по тикерам

    def _execute_trading_decisions(self, signals: List[Dict],
                                   prices: Dict[str, float], securities: Dict):
        """Исполнение торговых решений с RL"""
        executed_count = 0

        for signal in signals[:5]:
            ticker = signal['ticker']
            action_str = signal['action']  # 'BUY', 'SELL'
            confidence = signal['confidence']

            if ticker not in prices:
                continue

            price = prices[ticker]
            action_idx = {'BUY': 0, 'HOLD': 1, 'SELL': 2}[action_str]

            # 1. Получаем или создаем состояние для тикера
            if ticker in self.ticker_states:
                current_state = self.ticker_states[ticker]
            else:
                # Создаем начальное состояние
                current_state = self._create_initial_state(ticker, price, securities.get(ticker, {}))
                self.ticker_states[ticker] = current_state

            if action_str == 'BUY':
                # 2. Выбор стратегии для покупки
                strategy, stop_loss, take_profit = self._select_buy_strategy(
                    ticker, price, confidence, current_state
                )

                # 3. Исполнение
                quantity, risk_amount = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    price=price,
                    stop_loss=stop_loss,
                    confidence=confidence,
                    strategy_multiplier=self.model.strategies[strategy]['risk_multiplier']
                )

                if quantity > 0 and self.portfolio.buy(ticker, quantity, price):
                    # 4. Запись RL-опыта (состояние -> действие)
                    self._record_rl_experience(
                        ticker=ticker,
                        state=current_state,
                        action=action_idx,
                        strategy=strategy,
                        price=price,
                        quantity=quantity
                    )

                    # 5. Обновляем позицию информацией о стратегии
                    self.portfolio.positions[ticker]['strategy'] = strategy
                    self.portfolio.positions[ticker]['stop_loss'] = stop_loss
                    self.portfolio.positions[ticker]['take_profit'] = take_profit
                    self.portfolio.positions[ticker]['entry_state'] = current_state.cpu().numpy().tolist()

                    executed_count += 1

            elif action_str == 'SELL':
                if ticker in self.portfolio.positions:
                    # 6. Продажа - завершение RL-опыта
                    self._complete_rl_experience(ticker, price)

                    # Исполнение продажи
                    pos = self.portfolio.positions[ticker]
                    qty = pos['qty'] // 2 if pos['qty'] > 1 else pos['qty']

                    if qty > 0 and self.portfolio.sell(ticker, qty, price):
                        # Обновляем стратегию
                        strategy = pos.get('strategy', 'balanced')
                        self.model.record_strategy_outcome(
                            strategy_name=strategy,
                            action='SELL',
                            pnl=(price - pos['avg_price']) * qty,
                            hold_time=time.time() - pos.get('buy_time', time.time())
                        )

                        executed_count += 1

    def _record_rl_experience(self, ticker: str, state: torch.Tensor,
                              action: int, strategy: str, price: float, quantity: int):
        """Запись начала RL-опыта"""
        experience = {
            'ticker': ticker,
            'start_state': state.cpu(),
            'action': action,
            'strategy': strategy,
            'entry_price': price,
            'entry_time': time.time(),
            'quantity': quantity,
            'completed': False
        }

        self.pending_experiences.append(experience)

        # Сохраняем для отслеживания
        if ticker not in self.strategy_tracker:
            self.strategy_tracker[ticker] = []

        self.strategy_tracker[ticker].append({
            'strategy': strategy,
            'action': ['BUY', 'HOLD', 'SELL'][action],
            'entry_time': datetime.now().isoformat(),
            'entry_price': price
        })

    def _complete_rl_experience(self, ticker: str, exit_price: float):
        """Завершение RL-опыта при продаже"""
        # Находим незавершенный опыт для этого тикера
        for exp in self.pending_experiences:
            if exp['ticker'] == ticker and not exp['completed']:
                # Рассчитываем reward
                pnl = (exit_price - exp['entry_price']) * exp['quantity']
                hold_time = (time.time() - exp['entry_time']) / 3600  # в часах

                # Создаем следующее состояние
                next_state = self._create_next_state(ticker, exit_price)

                # Нормализованный reward
                reward = self._calculate_reward(pnl, hold_time, exp['strategy'])

                # Сохраняем опыт для обучения
                self.model.remember_experience(
                    state=exp['start_state'],
                    action=exp['action'],
                    reward=reward,
                    next_state=next_state,
                    done=True,
                    strategy=exp['strategy']
                )

                exp['completed'] = True
                break

    def _calculate_reward(self, pnl: float, hold_time: float, strategy: str) -> float:
        """Расчет награды с учетом стратегии"""
        base_reward = pnl / 100  # Нормализация

        # Штрафы/бонусы за время удержания
        target_hold_time = self.model.strategies[strategy].get('target_hold_time', 6)
        time_penalty = abs(hold_time - target_hold_time) * 0.1

        # Штраф за убыток
        loss_penalty = 0 if pnl > 0 else abs(pnl) * 0.2

        # Бонус за прибыль
        profit_bonus = pnl * 0.3 if pnl > 0 else 0

        final_reward = base_reward - time_penalty - loss_penalty + profit_bonus

        return final_reward

    def _create_initial_state(self, ticker: str, price: float, security_info: Dict) -> torch.Tensor:
        """Создание начального состояния для RL"""
        # Используем существующий метод модели
        momentum = security_info.get('momentum', 0.0)
        sentiment = self.news_core.get_current_sentiment(ticker)

        # Получаем технические данные
        indicators = self.technical_core.calculate_indicators(ticker)

        # Получаем новости
        news_items = self.rss_fetcher.get_news_for_ticker(ticker, limit=3)
        news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
        news_features = self.model.encode_news(news_texts)

        # Строим состояние
        state = self.model.build_state_vector(
            ticker=ticker,
            price=price,
            momentum=momentum,
            sentiment=sentiment,
            news_features=news_features,
            market_data={
                'volume': indicators.get('volume_ratio', 1.0),
                'spread': 0.01,
                'rsi': indicators.get('rsi', 50),
                'volatility': indicators.get('atr', 0) / price if price > 0 else 0.1,
                'sma_10_ratio': indicators.get('sma_10', price) / price if price > 0 else 1.0,
                'sma_20_ratio': indicators.get('sma_20', price) / price if price > 0 else 1.0,
                'bb_position': (price - indicators.get('bb_lower', price)) /
                               (indicators.get('bb_upper', price * 1.1) - indicators.get('bb_lower', price * 0.9))
                if indicators.get('bb_upper', price * 1.1) > indicators.get('bb_lower', price * 0.9) else 0.5
            }
        )

        return state

    def _select_buy_strategy(self, ticker: str, price: float,
                             confidence: float, state: torch.Tensor) -> Tuple[str, float, float]:
        """Выбор стратегии для покупки"""
        # Контекст рынка
        market_context = {
            'market_sentiment': self.model.market_sentiment,
            'volatility': self.model.volatility_index,
            'confidence': confidence,
            'time_of_day': datetime.now().hour / 24.0
        }

        # Используем модель для выбора стратегии
        action, strategy, strategy_confidence = self.model.choose_action_with_strategy(
            state=state,
            ticker=ticker,
            price=price,
            market_context=market_context
        )

        # Параметры стратегии
        strategy_params = self.model.strategies[strategy]

        # Расчет стоп-лосса и тейк-профита по стратегии
        if strategy == 'news_aggressive':
            stop_loss = price * (1 - 2.0 / 100)  # 2% стоп
            take_profit = price * (1 + 4.0 / 100)  # 4% тейк
        elif strategy == 'tech_conservative':
            stop_loss = price * (1 - 1.5 / 100)  # 1.5% стоп
            take_profit = price * (1 + 3.0 / 100)  # 3% тейк
        elif strategy == 'momentum':
            stop_loss = price * (1 - 1.0 / 100)  # 1% стоп
            take_profit = price * (1 + 2.0 / 100)  # 2% тейк
        else:  # balanced
            stop_loss = price * (1 - 2.5 / 100)  # 2.5% стоп
            take_profit = price * (1 + 5.0 / 100)  # 5% тейк

        return strategy, stop_loss, take_profit

    def _periodic_learning(self):
        """Периодическое обучение с RL"""
        try:
            # 1. Обучение на опыте
            if len(self.model.memory) > 100:
                loss = self.model.learn_from_experience(batch_size=32)

                if loss is not None:
                    logger.info(f"Обучение RL: Loss={loss:.6f}, "
                                f"Опытов={len(self.model.memory)}, "
                                f"Стратегий={len(self.model.strategy_performance)}")

            # 2. Оценка эффективности стратегий
            self._evaluate_strategies()

            # 3. Адаптация стратегий (раз в 50 циклов)
            if self.cycle_count % 50 == 0:
                self._adapt_strategies()

        except Exception as e:
            logger.error(f"Ошибка периодического обучения: {e}")

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

    def _adapt_strategies(self):
        """Адаптация параметров стратегий на основе эффективности"""
        for strategy_name, perf in self.model.strategy_performance.items():
            if perf['total_trades'] >= 10:
                win_rate = perf['win_rate']
                avg_pnl = perf['avg_pnl']

                # Адаптируем risk multiplier на основе эффективности
                current_multiplier = self.model.strategies[strategy_name]['risk_multiplier']

                if win_rate > 0.6 and avg_pnl > 0:
                    # Успешная стратегия - увеличиваем риск
                    new_multiplier = min(current_multiplier * 1.1, 2.0)
                elif win_rate < 0.4 or avg_pnl < 0:
                    # Неуспешная стратегия - уменьшаем риск
                    new_multiplier = max(current_multiplier * 0.9, 0.5)
                else:
                    new_multiplier = current_multiplier

                self.model.strategies[strategy_name]['risk_multiplier'] = new_multiplier

                logger.debug(f"Адаптация {strategy_name}: "
                             f"risk_multiplier {current_multiplier:.2f} -> {new_multiplier:.2f}")

    def _initialize_components(self):
        """Инициализация всех компонентов"""
        # Устанавливаем начальные настройки для компонентов
        self.portfolio.initial_capital = self.settings.get("initial_capital_rub", 10000)
        self.portfolio.max_positions = self.settings.get("max_positions", 5)

        # Запуск непрерывного сбора новостей
        if self.settings.get("enable_news_core", True):
            self.news_core.start_continuous_fetching()

        # Загрузка состояния портфеля
        self._load_portfolio_state()

        # Запуск планировщика задач
        self.scheduler.schedule_daily_tasks(
            pre_market_callback=self.pre_session_analysis,
            market_open_callback=self.on_market_open,
            market_close_callback=self.on_market_close,
            post_market_callback=self.post_session_analysis
        )

        logger.info("SmartBroker: все компоненты инициализированы")

    def _load_portfolio_state(self):
        """Загрузка состояния портфеля из файла"""
        try:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

                # Восстанавливаем портфель
                self.portfolio.positions = state.get('positions', {})
                self.portfolio.cash = state.get('cash', self.settings["initial_capital_rub"])

                # Обновляем время покупки
                for ticker, pos in self.portfolio.positions.items():
                    if 'buy_time' not in pos:
                        pos['buy_time'] = time.time()

                logger.info(f"Загружено состояние портфеля: {len(self.portfolio.positions)} позиций, "
                            f"{self.portfolio.cash:,.0f}₽ кэша")

        except Exception as e:
            logger.warning(f"Не удалось загрузить состояние портфеля: {e}")
            # Создаем начальное состояние
            self._save_portfolio_state()

    def _save_portfolio_state(self):
        """Сохранение состояния портфеля"""
        try:
            state = {
                'total_value': self.portfolio.get_total_value({}),
                'cash': self.portfolio.cash,
                'positions': self.portfolio.positions,
                'last_update': datetime.now().isoformat(),
                'initial_capital': self.settings["initial_capital_rub"]
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

        # Сброс дневных метрик риска
        self.risk_manager.reset_daily_metrics()

        # Анализ новостей
        self.news_core.fetch_all_news()

        # Получение списка ликвидных бумаг
        securities = self.moex.get_all_securities()
        logger.info(f"Загружено бумаг для анализа: {len(securities)}")

        # Обновление рыночного настроения
        self.analyze_sentiment(force=True)

        logger.info("Предсессионный анализ завершен")

    def on_market_open(self):
        """Действия при открытии рынка"""
        logger.info("РЫНОК ОТКРЫТ")
        self.trading_enabled = True

        # Проверка стоп-лоссов и тейк-профитов
        prices = self._get_current_prices()
        if prices:
            self.check_stops_and_tp(prices)

    def on_market_close(self):
        """Действия при закрытии рынка"""
        logger.info("РЫНОК ЗАКРЫТ")
        self.trading_enabled = False

        # Сохранение состояния
        self._save_portfolio_state()
        self.model.save_model()

        # Генерация дневного отчета
        self._generate_daily_report()

    def post_session_analysis(self):
        """Послерыночный анализ"""
        logger.info("ПОСЛЕРЫНОЧНЫЙ АНАЛИЗ")

        # Анализ результатов дня
        total_value = self.portfolio.get_total_value({})
        initial = self.settings["initial_capital_rub"]
        daily_pnl = total_value - self.portfolio.cash - sum(
            p['qty'] * p['avg_price'] for p in self.portfolio.positions.values()
        )

        logger.info(f"Итоги дня: Капитал {total_value:,.0f}₽, PnL: {daily_pnl:+,.0f}₽")
        logger.info(f"С начала: {((total_value / initial) - 1) * 100:+.2f}%")

    def run_cycle(self):
        """Основной торговый цикл"""
        if not self.trading_enabled or not self.scheduler.is_trading_time():
            return

        cycle_start = time.time()
        self.cycle_count += 1

        try:
            logger.debug(f"=== Торговый цикл #{self.cycle_count} ===")

            # 1. Получение данных
            securities = self.moex.get_all_securities()
            if not securities:
                logger.warning("Не удалось получить список бумаг")
                return

            # Берем топ-N по объему
            top_n = 120
            tickers = sorted(securities.items(),
                             key=lambda x: x[1].get('volume', 0),
                             reverse=True)[:top_n]
            tickers = [t[0] for t in tickers]
            self.current_tickers = tickers

            # 2. Получение текущих цен
            prices = {}
            for ticker in tickers:
                price = self.moex.get_price(ticker)
                if price:
                    prices[ticker] = price

            if len(prices) < 10:
                logger.warning(f"Слишком мало цен: {len(prices)}")
                return

            # 3. Генерация сигналов от всех ядер
            all_signals = []

            # Новостные сигналы
            if self.settings.get("enable_news_core", True):
                news_signals = self.news_core.generate_trading_signals(prices)
                all_signals.extend(news_signals)
                logger.debug(f"Сгенерировано новостных сигналов: {len(news_signals)}")

            # Технические сигналы
            if self.settings.get("enable_technical_core", True):
                tech_signals = self.technical_core.analyze_all_tickers(prices)
                all_signals.extend(tech_signals)
                logger.debug(f"Сгенерировано технических сигналов: {len(tech_signals)}")

            # 4. Агрегация и фильтрация сигналов
            filtered_signals = self._aggregate_signals(all_signals)
            self.signals_cache = filtered_signals[:10]  # Кэшируем топ-10

            # 5. Проверка стоп-лоссов и тейк-профитов
            self.check_stops_and_tp(prices)

            # 6. Исполнение торговых решений
            if filtered_signals and self.risk_manager.check_daily_limits():
                self._execute_trading_decisions(filtered_signals, prices, securities)

            # 7. Ребалансировка портфеля
            if self.cycle_count % 5 == 0:  # Каждые 5 циклов
                self._rebalance_portfolio(prices, securities)

            # 8. Периодическое обучение модели
            if self.cycle_count % 10 == 0:
                self._periodic_learning()

            # 9. Сохранение состояния
            if self.cycle_count % 20 == 0:
                self._save_portfolio_state()

            # Лог цикла
            cycle_time = time.time() - cycle_start
            total_value = self.portfolio.get_total_value(prices)

            logger.info(f"Цикл #{self.cycle_count} завершен за {cycle_time:.1f}с | "
                        f"Портфель: {total_value:,.0f}₽ | "
                        f"Позиций: {len(self.portfolio.positions)} | "
                        f"Сигналов: {len(filtered_signals)}")

        except Exception as e:
            logger.error(f"Критическая ошибка в run_cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

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

        # Группируем по тикерам
        ticker_signals = defaultdict(list)
        for signal in signals:
            ticker = signal['ticker']
            ticker_signals[ticker].append(signal)

        # Агрегируем сигналы для каждого тикера
        aggregated = []

        for ticker, sig_list in ticker_signals.items():
            # Подсчитываем голоса за каждое действие
            action_scores = defaultdict(float)

            for sig in sig_list:
                action = sig['action']
                confidence = sig.get('confidence', 0.5)

                # Взвешиваем по уверенности и типу сигнала
                weight = confidence
                if sig.get('reason') == 'news_analysis':
                    weight *= 1.2  # Новостные сигналы важнее

                action_scores[action] += weight

            # Выбираем действие с максимальным score
            if action_scores:
                best_action = max(action_scores.items(), key=lambda x: x[1])

                # Только если уверенность выше порога
                if best_action[1] > self.settings.get('model_confidence_threshold', 0.65):
                    aggregated_signal = {
                        'ticker': ticker,
                        'action': best_action[0],
                        'confidence': best_action[1] / len(sig_list),  # Нормализуем
                        'source_count': len(sig_list),
                        'sources': [s.get('reason', 'unknown') for s in sig_list],
                        'timestamp': datetime.now().isoformat()
                    }
                    aggregated.append(aggregated_signal)

        # Сортируем по уверенности
        aggregated.sort(key=lambda x: x['confidence'], reverse=True)

        return aggregated

    def _execute_trading_decisions(self,
                                   signals: List[Dict],
                                   prices: Dict[str, float],
                                   securities: Dict):
        """Исполнение торговых решений"""
        executed_count = 0

        for signal in signals[:5]:  # Обрабатываем топ-5 сигналов
            ticker = signal['ticker']
            action = signal['action']
            confidence = signal['confidence']

            if ticker not in prices:
                continue

            price = prices[ticker]

            if action == 'BUY':
                # Проверяем, нет ли уже позиции
                if ticker in self.portfolio.positions:
                    continue

                # Расчет стоп-лосса (фиксированный процент)
                stop_loss = price * (1 - self.settings.get('stop_loss_percent', 3.0) / 100)

                # Расчет размера позиции через Risk Manager
                quantity, risk_amount = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    price=price,
                    stop_loss=stop_loss,
                    confidence=confidence
                )

                if quantity > 0:
                    # Покупка
                    if self.portfolio.buy(ticker, quantity, price):
                        # Обновляем время покупки
                        self.portfolio.positions[ticker]['buy_time'] = time.time()

                        # Обновляем Risk Manager
                        self.risk_manager.update_trade_result(
                            ticker=ticker,
                            action='BUY',
                            quantity=quantity,
                            price=price,
                            pnl=0.0
                        )

                        # Запись для обучения модели
                        self._record_trade_for_learning(ticker, 'BUY', price, quantity, confidence)

                        executed_count += 1
                        logger.info(f"Исполнено BUY: {ticker} {quantity} @ {price:.2f} "
                                    f"(риск: {risk_amount:.0f}₽, conf: {confidence:.2f})")

            elif action == 'SELL':
                # Продажа только если есть позиция
                if ticker in self.portfolio.positions:
                    pos = self.portfolio.positions[ticker]
                    qty = pos['qty']

                    if qty > 0:
                        # Продаем половину позиции
                        sell_qty = qty // 2
                        if sell_qty > 0:
                            if self.portfolio.sell(ticker, sell_qty, price):
                                # Расчет PnL
                                entry_price = pos['avg_price']
                                pnl = (price - entry_price) * sell_qty

                                # Обновляем Risk Manager
                                self.risk_manager.update_trade_result(
                                    ticker=ticker,
                                    action='SELL',
                                    quantity=sell_qty,
                                    price=price,
                                    pnl=pnl
                                )

                                # Запись для обучения модели
                                self._record_trade_for_learning(ticker, 'SELL', price, sell_qty, confidence, pnl)

                                executed_count += 1
                                logger.info(f"Исполнено SELL: {ticker} {sell_qty} @ {price:.2f} "
                                            f"(PnL: {pnl:+.0f}₽)")

        if executed_count > 0:
            logger.info(f"Исполнено сделок: {executed_count}")

    def _record_trade_for_learning(self, ticker: str, action: str, price: float,
                                   quantity: int, confidence: float, pnl: float = 0.0):
        """Запись сделки для обучения модели"""
        try:
            # Получаем сентимент для тикера
            sentiment = 0.0
            if hasattr(self.news_core, 'get_current_sentiment'):
                sentiment = self.news_core.get_current_sentiment(ticker)

            # Записываем в модель
            if action == 'BUY':
                self.model.record_trade_outcome(
                    ticker=ticker,
                    action=action,
                    entry_price=price,
                    exit_price=price,  # При покупке exit = entry
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
                hold_time = (time.time() - pos.get('buy_time', time.time())) / 3600  # в часах

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
        """Проверка стоп-лоссов и тейк-профитов"""
        cfg = self.settings

        for ticker, pos in list(self.portfolio.positions.items()):
            if ticker not in prices:
                continue

            price = prices[ticker]
            entry_price = pos['avg_price']

            if entry_price <= 0:
                continue

            change_pct = (price - entry_price) / entry_price * 100
            hold_time = time.time() - pos.get('buy_time', time.time())

            # Стоп-лосс
            if change_pct <= -cfg.get('stop_loss_percent', 3.0):
                qty = pos['qty']
                if self.portfolio.sell(ticker, qty, price):
                    pnl = (price - entry_price) * qty

                    self.risk_manager.update_trade_result(
                        ticker=ticker,
                        action='STOP_LOSS',
                        quantity=qty,
                        price=price,
                        pnl=pnl
                    )

                    logger.warning(f"СТОП-ЛОСС: {ticker} {qty} @ {price:.2f} "
                                   f"({change_pct:+.1f}%, PnL: {pnl:+.0f}₽)")

            # Тейк-профит
            elif change_pct >= cfg.get('take_profit_percent', 6.0):
                qty = pos['qty'] // 2  # Продаем половину
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
        """Ребалансировка портфеля"""
        current_count = len(self.portfolio.positions)
        target_count = self.settings.get('target_positions', 4)

        # Если позиций меньше целевого - докупаем
        if current_count < target_count and self.portfolio.cash > self.settings["min_cash_per_trade"]:
            # Используем ранжирование от модели
            ticker_sentiment = {t: 0.0 for t in prices.keys()}
            news_by_ticker = {t: [] for t in prices.keys()}

            candidates = self.model.rank_candidates(
                prices=prices,
                securities=securities,
                ticker_sentiment=ticker_sentiment,
                news_by_ticker=news_by_ticker
            )

            bought = 0
            for ticker, score in candidates:
                if bought >= 2:  # Макс 2 позиции за ребалансировку
                    break

                if ticker in self.portfolio.positions:
                    continue

                price = prices[ticker]
                max_qty = int(self.portfolio.cash * 0.15 / price)  # Макс 15% кэша

                if max_qty > 0:
                    qty = max(1, max_qty // 2)  # Берем половину от максимального

                    if self.portfolio.buy(ticker, qty, price):
                        self.portfolio.positions[ticker]['buy_time'] = time.time()
                        bought += 1

                        logger.info(f"Ребалансировка BUY: {ticker} {qty} @ {price:.2f} "
                                    f"(score: {score:.2f})")

    def _periodic_learning(self):
        """Периодическое обучение модели"""
        try:
            loss = self.model.periodic_learning()
            if loss is not None:
                logger.info(f"Обучение модели: Loss={loss:.6f}")
        except Exception as e:
            logger.error(f"Ошибка обучения модели: {e}")

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
                'market_sentiment': self.model.market_sentiment,
                'signals_generated': len(self.signals_cache),
                'risk_metrics': self.risk_manager.get_risk_metrics()
            }

            # Сохраняем отчет
            report_file = f"data/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Дневной отчет сохранен: {report_file}")

        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")

    def analyze_sentiment(self, force: bool = False):
        """Анализ рыночного настроения"""
        try:
            # Используем RSS фетчер для новостей
            news_items = self.rss_fetcher.fetch_all_news()

            if not news_items:
                return

            # Анализ тональности
            sentiments = self.news_core.analyze_news_sentiment(news_items)
            market_sentiment = sentiments.get('MARKET', 0.0)

            # Обновляем в модели
            self.model.update_market_sentiment(market_sentiment)

            logger.info(f"Рыночное настроение обновлено: {market_sentiment:+.3f}")

        except Exception as e:
            logger.error(f"Ошибка анализа сентимента: {e}")

    def update_settings(self, new_settings: Dict):
        """Обновление настроек брокера"""
        try:
            # Обновляем основные настройки
            self.settings.update(new_settings)

            # Специфичные обновления компонентов
            if 'initial_capital_rub' in new_settings:
                self.portfolio.initial_capital = new_settings['initial_capital_rub']

            if 'max_positions' in new_settings:
                self.portfolio.max_positions = new_settings['max_positions']

            # Обновление Risk Manager
            if 'risk_per_trade_percent' in new_settings:
                if hasattr(self.risk_manager, 'config'):
                    self.risk_manager.config['risk_per_trade_percent'] = new_settings['risk_per_trade_percent']

            # Обновление параметров стоп-лоссов
            stop_loss_updates = ['stop_loss_percent', 'take_profit_percent', 'max_position_weight_percent']
            for param in stop_loss_updates:
                if param in new_settings:
                    if hasattr(self.risk_manager, 'config') and param in self.risk_manager.config:
                        self.risk_manager.config[param] = new_settings[param]

            logger.info(f"Настройки брокера обновлены: {new_settings}")

        except Exception as e:
            logger.error(f"Ошибка обновления настроек брокера: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Получение сводки портфеля"""
        prices = self._get_current_prices()
        total_value = self.portfolio.get_total_value(prices)

        summary = {
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'positions_count': len(self.portfolio.positions),
            'initial_capital': self.settings["initial_capital_rub"],
            'total_pnl': total_value - self.settings["initial_capital_rub"],
            'pnl_percent': ((total_value / self.settings["initial_capital_rub"]) - 1) * 100,
            'current_signals': self.signals_cache[:5],
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'session_info': self.scheduler.get_session_info(),
            'last_update': datetime.now().isoformat()
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
                'weight': (position_value / total_value * 100) if total_value > 0 else 0
            })

        summary['positions'] = positions_detail

        return summary

    def update_settings(self, new_settings: Dict):
        """Обновление настроек"""
        self.settings.update(new_settings)
        logger.info(f"Настройки обновлены: {new_settings}")

    def shutdown(self):
        """Корректное завершение работы"""
        logger.info("Завершение работы SmartBroker...")

        # Останавливаем сбор новостей
        self.news_core.stop_continuous_fetching()

        # Сохраняем состояние
        self._save_portfolio_state()
        self.model.save_model()

        logger.info("SmartBroker завершил работу")