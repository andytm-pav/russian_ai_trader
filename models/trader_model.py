"""
Модернизированная модель трейдера с улучшенной архитектурой
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
from utils.logger import get_logger

logger = get_logger("TRADER_MODEL")
warnings.filterwarnings('ignore')


class PrioritizedReplayBuffer:
    """Буфер с приоритетами для важных опытов (TD-error)"""

    def __init__(self, max_size=5000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.position = 0
        self.size = 0

    def add(self, experience, td_error=None):
        if td_error is None:
            priority = self.priorities.max() if self.size > 0 else 1.0
        else:
            priority = (abs(td_error) + 1e-6) ** self.alpha

        if self.size < self.max_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        if self.size < batch_size:
            return None, None, None

        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha


# ========== КОНСТАНТЫ МОДЕЛИ (ВЫНЕСЕНЫ В КОНФИГ) ==========
# Загружаем из конфига при инициализации
DEFAULT_CONFIG = {
    "news_embedding_dim": 312,
    "news_encoded_dim": 132,
    "base_state_dim": 210,
    "strategy_params_dim": 6,
    "total_state_dim": 216,

    "news_encoder_hidden_dim": 256,
    "news_encoder_intermediate_dim": 512,
    "policy_hidden_dim_1": 256,
    "policy_hidden_dim_2": 128,
    "policy_hidden_dim_3": 64,
    "policy_hidden_dim_4": 32,

    "learning_rate": 0.0005,
    "gamma": 0.95,
    "memory_size": 5000,
    "batch_size": 32,
    "min_experiences_for_learning": 100,

    "dropout_rate_1": 0.2,
    "dropout_rate_2": 0.3,
    "weight_decay": 0.01,
    "gradient_clip_value": 0.5,
    "entropy_bonus_coeff": 0.01,

    "exploration_rate": 0.3,
    "confidence_boost_factor": 0.4,
    "action_exploration_rate": 0.2,
    "strategy_memory_size": 2000,
    "min_trades_for_evaluation": 10,

    "max_sentiment_history": 200,
    "volatility_scaling_factor": 3.0,
    "base_volatility": 0.7,
    "sentiment_smoothing_alpha": 0.1,
    "risk_base": 0.4,
    "max_risk": 0.85,
    "min_risk": 0.1,

    "profit_threshold": 0.05,
    "loss_threshold": -0.03,
    "significant_loss": -0.02,
    "small_profit": 0.01,
    "min_hold_time": 0.5,

    "price_normalization": 10000.0,
    "volume_normalization": 1e7,
    "momentum_scaling": 20.0,
    "volatility_scaling": 10.0,
    "pe_normalization": 100.0,
    "rsi_normalization": 100.0,

    "top_candidates_limit": 30,
    "max_trades_for_experience": 50,
    "max_hold_time_days": 7.0,

    "auto_save_interval": 50
}


class NewsEncoder(nn.Module):
    """Улучшенный энкодер новостей"""

    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, DEFAULT_CONFIG["news_encoder_intermediate_dim"]),
            nn.LayerNorm(DEFAULT_CONFIG["news_encoder_intermediate_dim"]),
            nn.ReLU(),
            nn.Dropout(DEFAULT_CONFIG["dropout_rate_1"]),

            nn.Linear(DEFAULT_CONFIG["news_encoder_intermediate_dim"], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DEFAULT_CONFIG["dropout_rate_1"]),

            nn.Linear(hidden_dim, encoded_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class TradingPolicyNetwork(nn.Module):
    """Политика с предсказанием цены"""

    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, DEFAULT_CONFIG["policy_hidden_dim_1"]),
            nn.LayerNorm(DEFAULT_CONFIG["policy_hidden_dim_1"]),
            nn.ReLU(),
            nn.Dropout(DEFAULT_CONFIG["dropout_rate_2"]),

            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_1"], DEFAULT_CONFIG["policy_hidden_dim_2"]),
            nn.LayerNorm(DEFAULT_CONFIG["policy_hidden_dim_2"]),
            nn.ReLU(),
            nn.Dropout(DEFAULT_CONFIG["dropout_rate_2"]),

            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_2"], DEFAULT_CONFIG["policy_hidden_dim_3"]),
            nn.ReLU()
        )

        self.action_net = nn.Sequential(
            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_3"], DEFAULT_CONFIG["policy_hidden_dim_4"]),
            nn.ReLU(),
            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_4"], action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_3"], DEFAULT_CONFIG["policy_hidden_dim_4"]),
            nn.ReLU(),
            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_4"], 1)
        )

        self.predictor = nn.Sequential(
            nn.Linear(DEFAULT_CONFIG["policy_hidden_dim_3"], 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, state, news_features=None):
        state_features = self.state_net(state)

        action_probs = self.action_net(state_features)
        state_value = self.value_net(state_features)
        price_pred = self.predictor(state_features)

        return action_probs, state_value, price_pred


class AdvancedTraderModel:
    """Продвинутая модель трейдера"""

    def __init__(self, model_dir: str = "models/saved_trader"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка конфигов
        self.rl_config = self._load_rl_config()
        self.strategy_config = self._load_strategy_config()
        self.memory_config = self._load_memory_config()

        # Загрузка весов модели и нормализации из конфига
        self.model_weights = self.rl_config.get('model_weights', {})
        self.normalization = self.rl_config.get('normalization', {})

        # Загрузка BERT конфига
        self.bert_config = self.rl_config.get('bert_config', {
            'max_length': 128,
            'model_name': 'cointegrated/rubert-tiny2'
        })

        # Загрузка расписания торгов
        try:
            with open("config/market_schedule.json", "r", encoding="utf-8") as f:
                market_schedule = json.load(f)

            sessions = market_schedule.get('sessions', {})
            main_session = sessions.get('main_session', {})
            evening_session = sessions.get('evening_session', {})

            # Основная сессия
            main_end = main_session.get('end', '18:50')
            main_h, main_m = map(int, main_end.split(':'))
            self.main_session_close = main_h + main_m / 60.0

            # Вечерняя сессия
            evening_start = evening_session.get('start', '19:00')
            evening_h, evening_m = map(int, evening_start.split(':'))
            self.evening_session_start = evening_h + evening_m / 60.0
        except Exception as e:
            logger.warning(f"Не удалось загрузить market_schedule.json: {e}")
            self.main_session_close = 18.833
            self.evening_session_start = 19.0

        # Загрузка констант для комиссий из normalization
        self.commission_rate = self.normalization.get('commission_rate', 0.003)
        self.max_trades_per_hour = self.normalization.get('max_trades_per_hour', 10)
        self.initial_capital = self.normalization.get('initial_capital', 10000.0)
        self.commission_normalization = self.normalization.get('commission_normalization', 1000.0)

        # Загрузка констант для нормализации макро-признаков
        self.imoex_normalization = self.normalization.get('imoex_normalization', 3000.0)
        self.moexfn_normalization = self.normalization.get('moexfn_normalization', 10000.0)
        self.brent_normalization = self.normalization.get('brent_normalization', 100.0)

        # Загрузка времени закрытия из rl_config
        self.close_hour = self.rl_config.get('close_hour', 18.833)
        self.evening_start = self.rl_config.get('evening_start', 19.0)

        # Параметры из конфигов
        state_params = self.rl_config["state_parameters"]
        self.news_encoded_dim = state_params["news_features_size"]
        self.base_state_dim = state_params["state_vector_size"]
        self.strategy_params_dim = state_params["strategy_params_size"]
        self.total_state_dim = state_params["total_state_size"]

        # Синхронизация глобальных констант для обратной совместимости
        global NEWS_ENCODED_DIM, BASE_STATE_DIM, TOTAL_STATE_DIM
        NEWS_ENCODED_DIM = self.news_encoded_dim
        BASE_STATE_DIM = self.base_state_dim
        TOTAL_STATE_DIM = self.total_state_dim

        # Загрузка параметров построения состояния
        self.state_building = self.rl_config.get('state_building', {
            'sector_onehot_size': 5,
            'strategy_onehot_size': 8,
            'default_stop_loss_mult': 0.97,
            'default_take_profit_mult': 1.05,
            'position_size_fraction': 0.5,
            'breakeven_multiplier': 1.006,
            'drawdown_multiplier': 2.0
        })

        # Загрузка размерностей состояния
        self.state_dimensions = self.rl_config.get('state_dimensions', {
            'price_volume': 4,
            'technical': 7,
            'news': 134,
            'fundamental': 7,
            'position': 5,
            'portfolio': 5,
            'risk': 3,
            'time': 4,
            'strategy': 11,
            'macro_extra': 4,
            'commission': 8
        })

        # Загрузка сентимент конфига
        self.sentiment_keywords = self.rl_config.get('sentiment_config', {
            'positive_keywords': ["рост", "прибыль", "увеличение", "дивиденд", "выше", "улучшение", "рекомендуют",
                                  "покупать"],
            'negative_keywords': ["падение", "убыток", "снижение", "проблемы", "ниже", "сокращение", "продавать",
                                  "снижают"],
            'market_keywords': ["рынок", "акци", "бирж", "фондов", "инвест", "торг", "ликвид", "волатиль"],
            'sentiment_divisor': 10,
            'noise_scale': 0.1
        })

        # Загрузка параметров приоритетного буфера
        self.prioritized_buffer_config = self.rl_config.get('prioritized_buffer', {
            'alpha': 0.6,
            'beta': 0.4,
            'beta_increment': 0.001
        })

        # Загрузка статистик по умолчанию
        default_stats = self.rl_config.get('default_stats', {})

        default_strategy_perf = default_stats.get('strategy_performance', {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.5
        })

        default_error_memory = default_stats.get('error_memory', {
            'failed_trades': [],
            'avg_loss': 0.0,
            'last_failure': None,
            'failure_count': 0,
            'success_rate': 0.5,
            'total_trades': 0
        })

        default_ticker_stats = default_stats.get('ticker_stats', {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0,
            'success_rate': 0.5,
            'last_trade': None
        })

        # Загрузка начального рыночного состояния
        initial_market = self.rl_config.get('initial_market_state', {
            'market_sentiment': 0.0,
            'volatility_index': 1.0
        })

        # Количество действий (BUY/HOLD/SELL)
        self.action_dim = self.rl_config.get('action_dim', 3)

        # Расчет ожидаемой размерности
        self.expected_dim = sum(self.state_dimensions.values()) + self.normalization.get('reserved_slots', 10)

        # Загрузка BERT
        self.bert_model, self.bert_tokenizer = self._load_bert_model()

        # Инициализация сетей
        self.news_encoder = NewsEncoder(
            input_dim=self.normalization.get('news_embedding_dim', 312),
            hidden_dim=self.model_weights.get('news_encoder_hidden_dim', 256),
            encoded_dim=self.news_encoded_dim
        ).to(self.device)

        self.policy_net = TradingPolicyNetwork(
            state_dim=self.total_state_dim,
            action_dim=self.action_dim
        ).to(self.device)

        # Оптимизаторы
        self.policy_optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.rl_config.get("learning_rate", 0.0005),
            weight_decay=self.model_weights.get('weight_decay', 0.01)
        )

        # Память
        self.memory = deque(maxlen=self.rl_config.get("memory_size", 5000))
        self.prioritized_buffer = PrioritizedReplayBuffer(
            max_size=self.rl_config.get("memory_size", 5000),
            alpha=self.prioritized_buffer_config.get('alpha', 0.6),
            beta=self.prioritized_buffer_config.get('beta', 0.4),
            beta_increment=self.prioritized_buffer_config.get('beta_increment', 0.001)
        )

        # Статистика
        self.strategy_performance = defaultdict(lambda: default_strategy_perf.copy())
        self.strategies = self.strategy_config['strategies']
        self.error_memory = defaultdict(lambda: default_error_memory.copy())
        self.ticker_stats = defaultdict(lambda: default_ticker_stats.copy())

        # Рыночное состояние
        self.market_sentiment = initial_market.get('market_sentiment', 0.0)
        self.sentiment_history = deque(maxlen=self.model_weights.get('max_sentiment_history', 200))
        self.volatility_index = initial_market.get('volatility_index', 1.0)

        # Параметры обучения
        self.gamma = self.rl_config.get("gamma", 0.95)

        strategy_selection = self.strategy_config.get('strategy_selection', {})
        self.exploration_rate = strategy_selection.get('exploration_rate', 0.3)
        self.confidence_boost_factor = strategy_selection.get('confidence_boost_factor', 0.4)

        # Загрузка сохраненной модели
        self.load_model()
        self.load_memory()

        print(f"[TraderModel] Инициализирована на {self.device}")
        print(
            f"[TraderModel] Размерность состояния: {self.base_state_dim} + {self.strategy_params_dim} = {self.total_state_dim}")
        print(f"[TraderModel] Конфиг: {self.rl_config.get('state_parameters', {})}")

    def _load_rl_config(self) -> Dict:
        """Загрузка RL конфига"""
        try:
            with open("config/rl_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки RL конфига: {e}")
            return {
                "state_parameters": {
                    "state_vector_size": 181,
                    "total_state_size": 187,
                    "news_features_size": 128,
                    "strategy_params_size": 6,
                    "market_features_count": 38,
                    "reserved_slots": 10
                },
                "learning_rate": 0.0005,
                "gamma": 0.95,
                "memory_size": 5000,
                "batch_size": 32
            }

    def _load_strategy_config(self) -> Dict:
        """Загрузка конфигурации стратегий"""
        try:
            with open("config/strategies.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки конфига стратегий: {e}")
            return {
                'strategies': {
                    'balanced': {
                        'news_weight': 0.5,
                        'tech_weight': 0.5,
                        'risk_multiplier': 1.0,
                        'target_hold_time_hours': 6,
                        'stop_loss_percent': 2.5,
                        'take_profit_percent': 5.0
                    }
                },
                'strategy_selection': {
                    'exploration_rate': 0.3,
                    'confidence_boost_factor': 0.4,
                    'memory_size': 2000,
                    'min_trades_for_evaluation': 10
                }
            }

    def _load_memory_config(self) -> Dict:
        """Загрузка конфигурации памяти"""
        try:
            with open("config/rl_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get('memory_serialization', {
                    'enable_autosave': True,
                    'autosave_interval': 10,
                    'memory_file': 'models/saved_trader/memory_buffer.pkl',
                    'max_memory_to_save': 5000,
                    'compression': True
                })
        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки конфига памяти: {e}")
            return {
                'enable_autosave': True,
                'autosave_interval': 10,
                'memory_file': 'models/saved_trader/memory_buffer.pkl',
                'max_memory_to_save': 5000,
                'compression': True
            }

    def _load_bert_model(self):
        """Загрузка BERT модели"""
        try:
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
            model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
            model.to(self.device)
            model.eval()
            print("[TraderModel] ✓ Загружена модель RuBERT-tiny2")
            return model, tokenizer
        except Exception as e:
            print(f"[TraderModel] ⚠ Не удалось загрузить BERT: {e}")
            return None, None

    def build_state_vector(self,
                           ticker: str,
                           price: float,
                           momentum: float,
                           sentiment: float,
                           news_features: torch.Tensor,
                           market_data: Dict,
                           market_sentiment: float = 0.0,
                           portfolio=None) -> torch.Tensor:
        """
        Построение вектора состояния (210 признаков)
        """
        # Используем self.normalization, а не norm
        norm = self.normalization
        # kwargs = {'portfolio': portfolio} if portfolio is not None else {}

        # Новости
        if news_features.numel() > 0 and news_features.shape[1] == self.news_encoded_dim:
            news_vec = news_features.mean(dim=0).cpu().numpy()
        else:
            news_vec = np.zeros(self.news_encoded_dim)

        # Статистика тикера
        stats = self.ticker_stats[ticker]
        success_rate = stats['success_rate']
        total_trades = min(stats['total_trades'] / norm.get('max_trades_for_experience', 50), 2.0)
        avg_hold_time = min(stats['avg_hold_time'] / norm.get('hours_in_day', 24.0),
                            norm.get('max_hold_time_days_norm', 7.0))

        # Риск
        risk_score = self.calculate_risk_score(ticker, price, sentiment)

        # === 1. Цена/Объем (4) ===
        features = [
            price / norm.get('price_normalization', 10000.0),
            market_data.get('volume', 0) / norm.get('volume_normalization', 1e7),
            market_data.get('spread', 0.01) * 100,
            market_data.get('market_cap', 0) / norm.get('market_cap_divisor', 1e12),
        ]

        # === 2. Технические (7) ===
        features.extend([
            market_data.get('rsi', 50) / norm.get('rsi_normalization', 100.0),
            market_data.get('sma_10_ratio', 1.0),
            market_data.get('sma_20_ratio', 1.0),
            market_data.get('bb_position', 0.5),
            market_data.get('atr', 0) / price if price > 0 else 0.1,
            market_data.get('volume_ratio', 1.0),
            momentum * norm.get('momentum_scaling', 20.0),
        ])

        # === 3. Новости (134) ===
        features.extend(news_vec.tolist())
        features.extend([0.0] * 2)  # резерв (2 слота)

        # === 4. Фундаментальные (7) ===
        sector_onehot = [0] * self.state_building.get('sector_onehot_size', 5)
        sector = market_data.get('sector', 'other')
        sector_map = {'финансы': 0, 'нефтегаз': 1, 'металлы': 2, 'телеком': 3, 'other': 4}
        if sector in sector_map:
            sector_onehot[sector_map[sector]] = 1

        features.extend(sector_onehot)
        features.extend([
            market_data.get('lot_size', 1) / norm.get('lot_size_divisor', 100.0),
            market_data.get('min_step', 0.01) * 100,
        ])

        # === 5. Позиция (5) ===
        has_position = 0.0
        position_pnl = 0.5
        dist_to_stop = 0.0
        dist_to_take = 0.0
        hold_ratio = 0.0

        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'positions'):
            if ticker in self.portfolio.positions:
                pos = self.portfolio.positions[ticker]
                entry = pos.get('avg_price', price)
                stop = pos.get('stop_loss', entry * self.state_building.get('default_stop_loss_mult', 0.97))
                take = pos.get('take_profit', entry * self.state_building.get('default_take_profit_mult', 1.05))

                has_position = 1.0
                pnl_raw = (price - entry) / entry
                position_pnl = max(-0.5, min(0.5, pnl_raw)) * 2 + 0.5

                if stop < entry:
                    dist_to_stop = (price - stop) / (entry - stop)
                    dist_to_stop = max(0.0, min(1.0, dist_to_stop))

                if take > entry:
                    dist_to_take = (take - price) / (take - entry)
                    dist_to_take = max(0.0, min(1.0, dist_to_take))

                if 'buy_time' in pos:
                    hold_hours = (time.time() - pos['buy_time']) / 3600
                    target = pos.get('target_hold_hours', 6)
                    hold_ratio = min(hold_hours / target, 2.0) / 2.0

        features.extend([has_position, position_pnl, dist_to_stop, dist_to_take, hold_ratio])

        # === 6. Портфель (5) ===
        positions_norm = 0.0
        exposure_norm = 0.0
        cash_ratio = 0.0
        drawdown = 0.0
        daily_pnl_norm = 0.0

        if portfolio is not None:
            if hasattr(portfolio, 'positions'):
                positions_norm = min(len(portfolio.positions) / self.normalization.get('max_positions_norm', 10.0), 1.0)

            if hasattr(portfolio, 'initial_capital') and portfolio.initial_capital > 0:
                positions_value = sum(
                    p['qty'] * p.get('avg_price', 0)
                    for p in portfolio.positions.values()
                )
                exposure_norm = min(positions_value / portfolio.initial_capital, 1.0)

                if hasattr(portfolio, 'cash') and hasattr(portfolio, 'reserved_cash'):
                    available = portfolio.cash - portfolio.reserved_cash
                    cash_ratio = available / portfolio.initial_capital

                if hasattr(self, 'peak_value'):
                    if hasattr(portfolio, 'get_total_value'):
                        current_value = portfolio.get_total_value({})
                        if current_value < self.peak_value:
                            drawdown = (self.peak_value - current_value) / self.peak_value
                        if current_value > self.peak_value:
                            self.peak_value = current_value
        else:
            # Fallback: нет портфеля - используем значения по умолчанию
            logger.debug("Портфель не передан в build_state_vector, использую значения по умолчанию")
            positions_norm = 0.0
            exposure_norm = 0.0
            cash_ratio = 1.0
            drawdown = 0.0
            daily_pnl_norm = 0.0

        drawdown_mult = self.state_building.get('drawdown_multiplier', 2.0)
        features.extend([positions_norm, exposure_norm, cash_ratio, min(drawdown * drawdown_mult, 1.0), daily_pnl_norm])

        # === 7. Риск (3) ===
        features.extend([
            risk_score,
            self.volatility_index,
            market_sentiment,
        ])

        # === 8. Время (4) ===
        now = datetime.now()
        close_hour = getattr(self, 'main_session_close', 18.833)
        evening_start = getattr(self, 'evening_session_start', 19.0)

        current_hour = now.hour + now.minute / 60.0
        time_to_close = max(0.0, (close_hour - current_hour) / 24.0)
        is_evening = 1.0 if current_hour >= evening_start else 0.0

        features.extend([
            now.hour / norm.get('hours_in_day', 24.0),
            now.weekday() / norm.get('days_in_week', 7.0),
            time_to_close,
            is_evening,
        ])

        # === 9. Стратегия (11) ===
        strategy_onehot = [0] * self.state_building.get('strategy_onehot_size', 8)
        current_strategy = getattr(self, 'current_strategy', 'balanced')
        strategy_list = list(self.strategies.keys())
        if current_strategy in strategy_list:
            strategy_onehot[strategy_list.index(current_strategy)] = 1

        features.extend(strategy_onehot)

        if current_strategy in self.strategy_performance:
            perf = self.strategy_performance[current_strategy]
            features.extend([
                perf['win_rate'],
                min(perf['total_trades'] / norm.get('max_trades_for_experience', 100.0), 1.0),
                max(-0.5, min(0.5, perf['avg_pnl'])) + 0.5,
            ])
        else:
            features.extend([0.5, 0.0, 0.5])

        # === 10. Дополнительные макро-признаки (4) ===
        imoex_norm = getattr(self, 'imoex_normalization', 3000.0)
        moexfn_norm = getattr(self, 'moexfn_normalization', 10000.0)
        brent_norm = getattr(self, 'brent_normalization', 100.0)

        macro_extra = [
            market_data.get('moexog', 0.0) / imoex_norm,
            market_data.get('moexfn', 0.0) / moexfn_norm,
            market_data.get('brent', 0.0) / brent_norm,
            market_data.get('brent_change', 0.0) / 100.0,
        ]
        features.extend(macro_extra)

        # === 11. Комиссионные издержки (8) ===
        commission_rate = getattr(self, 'commission_rate', 0.003)
        commission_reserve_ratio = 0.0
        commission_spent_ratio = 0.0
        avg_commission = 0.0
        commission_to_pnl = 0.0
        trade_frequency_penalty = 0.0
        expected_commission = 0.0
        breakeven_price_ratio = 1.0

        if portfolio is not None:
            # import time
            # datetime уже импортирован в начале файла

            initial_capital = getattr(self, 'initial_capital', 10000.0)
            commission_norm = getattr(self, 'commission_normalization', 1000.0)
            max_trades_per_hour = getattr(self, 'max_trades_per_hour', 10)

            commission_reserve_ratio = getattr(portfolio, 'commission_reserve', 0.0) / initial_capital
            commission_spent_ratio = getattr(portfolio, 'commission_spent_today', 0.0) / getattr(portfolio,
                                                                                                 'daily_commission_limit',
                                                                                                 100.0)

            total_commission = getattr(portfolio, 'total_commission', 0.0)
            total_trades = getattr(portfolio, 'total_trades', 1)
            avg_commission = total_commission / max(1, total_trades) / commission_norm

            total_pnl = getattr(portfolio, 'total_pnl', 0.0)
            commission_to_pnl = min(total_commission / max(1, abs(total_pnl)), 2.0) if total_pnl > 0 else 0.0

            trade_history = getattr(portfolio, 'trade_history', [])
            now = time.time()
            trades_last_hour = 0
            for t in trade_history:
                ts = t.get('timestamp')
                if ts:
                    if isinstance(ts, str):
                        try:
                            ts_float = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except:
                            continue
                    else:
                        ts_float = float(ts)
                    if ts_float > now - 3600:
                        trades_last_hour += 1

            trade_frequency_penalty = min(trades_last_hour / max_trades_per_hour, 1.0)

            max_positions = getattr(portfolio, 'max_positions', 10)
            position_value = (initial_capital / max_positions) * self.state_building.get('position_size_fraction', 0.5)
            expected_commission = (position_value * commission_rate) / initial_capital

            position = portfolio.positions.get(ticker) if hasattr(portfolio, 'positions') else None
            if position:
                entry_price = position.get('avg_price', price)
                breakeven_mult = self.state_building.get('breakeven_multiplier', 1.006)
                breakeven_price_ratio = (entry_price * breakeven_mult) / price if price > 0 else 1.0

        commission_features = [
            commission_reserve_ratio,
            commission_spent_ratio,
            avg_commission,
            commission_to_pnl,
            trade_frequency_penalty,
            expected_commission,
            commission_rate * 2,
            breakeven_price_ratio,
        ]

        features.extend(commission_features)

        # === 12. Резервные слоты ===
        feature_config = self.rl_config.get('feature_config', {})
        reserved_slots = feature_config.get('reserved_slots', 6)
        features.extend([0.0] * reserved_slots)

        # === 13. Дополнительные рыночные признаки ===
        market_feature_names = feature_config.get('market_features', [])
        for name in market_feature_names:
            features.append(market_data.get(name, 0.0))


        # === 14. Рыночная ликвидность и активность (2) ===
        features.append(market_data.get('market_liquidity_ratio', 0.0))
        features.append(market_data.get('market_activity_score', 0.0))



        # Проверка размерности
        expected_dim = self._get_expected_dimension()
        if len(features) != expected_dim:
            logger.warning(f"build_state_vector: ожидалось {expected_dim}, получено {len(features)}")
            if len(features) < expected_dim:
                features.extend([0.0] * (expected_dim - len(features)))
            else:
                features = features[:expected_dim]

        return torch.FloatTensor(features).to(self.device)

    def _get_expected_dimension(self) -> int:
        return self.base_state_dim



    def calculate_risk_score(self, ticker: str, price: float, sentiment: float) -> float:
        """Расчет риск-скора"""
        error_data = self.error_memory[ticker]
        stats = self.ticker_stats[ticker]
        norm = self.normalization  # <--- Добавлено!

        risk_config = self.rl_config.get('risk_calculation', {})
        success_threshold = norm.get('success_rate_threshold', 0.5)

        if error_data['failure_count'] == 0:
            base_risk = norm.get('risk_base', 0.4)
        else:
            failure_penalty = min(
                error_data['failure_count'] * risk_config.get('failure_penalty_rate', 0.08),
                risk_config.get('max_failure_penalty', 0.8)
            )
            loss_penalty = min(
                abs(error_data['avg_loss']) * risk_config.get('loss_penalty_multiplier', 1.5),
                risk_config.get('max_loss_penalty', 0.6)
            )
            success_bonus = max(stats['success_rate'] - success_threshold, 0) * risk_config.get('success_bonus_rate',
                                                                                                0.3)

            base_risk = norm.get('risk_base', 0.4) + failure_penalty + loss_penalty - success_bonus

        sentiment_factor = risk_config.get('base_sentiment_factor', 1.2) - abs(sentiment)
        volatility_factor = risk_config.get('base_volatility_factor', 0.8) + \
                            (self.volatility_index * risk_config.get('volatility_multiplier', 0.4))

        final_risk = base_risk * sentiment_factor * volatility_factor

        return max(norm.get('min_risk', 0.1), min(norm.get('max_risk', 0.85), final_risk))







    def get_price_pred_probs(self, price_pred):
        """Универсальное получение вероятностей из price_pred"""
        if price_pred.dim() == 1:
            # Вектор (3,)
            return torch.softmax(price_pred, dim=0).cpu().numpy()
        elif price_pred.dim() == 2 and price_pred.shape[0] == 1:
            # Матрица (1,3)
            return torch.softmax(price_pred, dim=1).cpu().numpy()[0]
        elif price_pred.dim() == 2:
            # Батч (batch_size, 3)
            return torch.softmax(price_pred, dim=1).cpu().numpy()
        else:
            logger.error(f"Неожиданная размерность price_pred: {price_pred.shape}")
            dummy = torch.zeros(3, device=price_pred.device)
            return torch.softmax(dummy, dim=0).cpu().numpy()

    def choose_action_with_strategy(self, state: torch.Tensor, ticker: str,
                                   price: float, market_context: Dict) -> Tuple[int, str, float]:
        """Выбор действия с учетом стратегии"""
        strategy_scores = {}

        # Если состояние базовое (без стратегии) - берем как есть
        if state.shape[-1] == self.base_state_dim:
            base_state = state
        else:
            base_state = state

        for strategy_name, params in self.strategies.items():
            # Добавляем параметры стратегии к состоянию
            strategy_state = self._create_strategy_state(base_state, params)

            with torch.no_grad():
                action_probs, state_value, price_pred = self.policy_net(strategy_state.unsqueeze(0))

            perf = self.strategy_performance[strategy_name]
            confidence_boost = perf['win_rate'] * self.confidence_boost_factor

            expected_value = state_value.item() + confidence_boost
            strategy_scores[strategy_name] = {
                'expected_value': expected_value,
                'action_probs': action_probs.cpu().numpy().flatten(),
                'params': params
            }

        # Выбор стратегии
        if np.random.random() < self.exploration_rate:
            chosen_strategy = np.random.choice(list(self.strategies.keys()))
        else:
            chosen_strategy = max(strategy_scores.items(),
                                key=lambda x: x[1]['expected_value'])[0]

        action_probs = strategy_scores[chosen_strategy]['action_probs']

        if np.random.random() < DEFAULT_CONFIG["action_exploration_rate"]:
            action = np.random.choice(len(action_probs))
        else:
            action = np.argmax(action_probs)

        confidence = action_probs[action]

        return action, chosen_strategy, confidence

    def _create_strategy_state(self, base_state: torch.Tensor,
                              strategy_params: Dict) -> torch.Tensor:
        """Добавление параметров стратегии к базовому состоянию"""
        if base_state.shape[-1] == self.total_state_dim:
            return base_state

        strategy_params_tensor = torch.tensor([
            float(strategy_params.get('news_weight', 0.5)),
            float(strategy_params.get('tech_weight', 0.5)),
            float(strategy_params.get('risk_multiplier', 1.0)),
            float(strategy_params.get('target_hold_time_hours', 6)) / 24.0,
            float(strategy_params.get('stop_loss_percent', 2.5)) / 100.0,
            float(strategy_params.get('take_profit_percent', 5.0)) / 100.0,
        ], dtype=torch.float32, device=self.device)

        return torch.cat([base_state, strategy_params_tensor])

    def get_state_value(self, state: torch.Tensor) -> float:
        """Получение оценки состояния"""
        try:
            self.policy_net.eval()
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                _, value, _ = self.policy_net(state)
                return value.item()
        except Exception as e:
            print(f"[TraderModel] Ошибка get_state_value: {e}")
            return 0.0

    def remember_experience(self, state: torch.Tensor, action: int, reward: float,
                            next_state: torch.Tensor, done: bool,
                            news_features: Optional[torch.Tensor] = None,  # ДОБАВИТЬ
                            td_error: Optional[float] = None,
                            pnl_rub: float = 0.0, sentiment_data=None):
        """Сохранение опыта"""
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        clip_min = self.rl_config.get('reward_clip_min', -0.5)
        clip_max = self.rl_config.get('reward_clip_max', 0.5)
        reward = max(clip_min, min(clip_max, reward))

        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'pnl_rub': pnl_rub,
            'next_state': next_state.cpu(),
            'done': done,
            'sentiment_data': sentiment_data,
            'timestamp': datetime.now().isoformat()
        }

        self.memory.append(experience)

        if hasattr(self, 'prioritized_buffer'):
            self.prioritized_buffer.add(experience, td_error)

        # Автосохранение
        if (self.memory_config['enable_autosave'] and
                len(self.memory) % self.memory_config['autosave_interval'] == 0):
            self.save_memory()

    def learn_from_experience(self, batch_size: int = 32):
        """Обучение на опыте"""
        if len(self.memory) < batch_size * 2:
            return None

        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        try:
            states = torch.stack([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

            self.policy_net.train()

            current_probs, current_values, _ = self.policy_net(states)

            with torch.no_grad():
                _, next_values, _ = self.policy_net(next_states)

            target_values = rewards + (1 - dones) * self.gamma * next_values

            value_loss = nn.SmoothL1Loss()(current_values, target_values.detach())

            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            entropy = dist.entropy().mean()
            entropy_bonus = DEFAULT_CONFIG["entropy_bonus_coeff"] * entropy

            total_loss = value_loss + policy_loss - entropy_bonus

            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), DEFAULT_CONFIG["gradient_clip_value"])
            self.policy_optimizer.step()

            self.policy_net.eval()

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка обучения: {e}")
            self.policy_net.eval()
            return None

    def learn_from_prioritized(self, batch_size: int = 32):
        """Обучение с приоритетной выборкой"""
        if not hasattr(self, 'prioritized_buffer') or self.prioritized_buffer.size < batch_size:
            return self.learn_from_experience(batch_size)

        batch, indices, weights = self.prioritized_buffer.sample(batch_size)

        try:
            states = torch.stack([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
            weights_tensor = torch.FloatTensor(weights).to(self.device)

            self.policy_net.train()

            current_probs, current_values, _ = self.policy_net(states)

            with torch.no_grad():
                _, next_values, _ = self.policy_net(next_states)

            target_values = rewards + (1 - dones) * self.gamma * next_values

            value_loss = (weights_tensor * nn.SmoothL1Loss(reduction='none')(
                current_values, target_values.detach())).mean()

            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(weights_tensor * log_probs * advantages).mean()

            entropy = dist.entropy().mean()
            entropy_bonus = DEFAULT_CONFIG["entropy_bonus_coeff"] * entropy

            total_loss = value_loss + policy_loss - entropy_bonus

            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), DEFAULT_CONFIG["gradient_clip_value"])
            self.policy_optimizer.step()

            self.policy_net.eval()

            td_errors = (target_values - current_values).detach().cpu().numpy().flatten()
            self.prioritized_buffer.update_priorities(indices, td_errors)

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка приоритетного обучения: {e}")
            self.policy_net.eval()
            return None

    def learn_from_experience_custom(self, states, actions, rewards, next_states, dones):
        """Обучение на готовом батче"""
        try:
            self.policy_net.train()

            current_probs, current_values, _ = self.policy_net(states)

            with torch.no_grad():
                _, next_values, _ = self.policy_net(next_states)

            target_values = rewards + (1 - dones) * self.gamma * next_values

            value_loss = nn.SmoothL1Loss()(current_values, target_values.detach())

            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            entropy = dist.entropy().mean()
            entropy_bonus = DEFAULT_CONFIG["entropy_bonus_coeff"] * entropy

            total_loss = value_loss + policy_loss - entropy_bonus

            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), DEFAULT_CONFIG["gradient_clip_value"])
            self.policy_optimizer.step()

            self.policy_net.eval()
            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка кастомного обучения: {e}")
            self.policy_net.eval()
            return None

    def encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Кодирование новостей"""
        if not news_texts:
            return torch.zeros(1, self.news_encoded_dim).to(self.device)

        if self.bert_model is not None and self.bert_tokenizer is not None:
            try:
                inputs = self.bert_tokenizer(
                    news_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.bert_config.get('max_length', 128),
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]

                self.news_encoder.eval()
                with torch.no_grad():
                    news_features = self.news_encoder(embeddings)

                return news_features

            except Exception as e:
                print(f"[TraderModel] Ошибка BERT кодирования: {e}")

        return self.simple_encode_news(news_texts)

    def simple_encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Упрощенное кодирование новостей"""
        norm = self.normalization  # <--- Добавлено!

        batch_size = len(news_texts)
        news_embedding_dim = norm.get('news_embedding_dim', 312)
        embeddings = torch.zeros(batch_size, news_embedding_dim).to(self.device)

        sentiment_dict = {
            'positive': self.sentiment_keywords.get('positive_keywords', ['рост', 'прибыль']),
            'negative': self.sentiment_keywords.get('negative_keywords', ['падение', 'убыток']),
            'market': self.sentiment_keywords.get('market_keywords', ['рынок', 'акци'])
        }

        max_length = norm.get('news_max_length', 1000)
        max_words = norm.get('news_max_words', 200)
        sentiment_divisor = self.sentiment_keywords.get('sentiment_divisor', 10)
        noise_scale = self.sentiment_keywords.get('noise_scale', 0.1)

        max_length = norm.get('news_max_length', 1000)
        max_words = norm.get('news_max_words', 200)
        sentiment_divisor = norm.get('sentiment_divisor', 10)
        noise_scale = norm.get('news_noise_scale', 0.1)

        for i, text in enumerate(news_texts):
            text_lower = text.lower()

            length = min(len(text) / max_length, 1.0)
            words = text_lower.split()
            word_count = min(len(words) / max_words, 1.0)
            unique_ratio = len(set(words)) / max(len(words), 1)

            pos_score = sum(1 for w in sentiment_dict['positive'] if w in text_lower) / sentiment_divisor
            neg_score = sum(1 for w in sentiment_dict['negative'] if w in text_lower) / sentiment_divisor
            market_score = sum(1 for w in sentiment_dict['market'] if w in text_lower) / sentiment_divisor

            pos_score = min(pos_score, 1.0)
            neg_score = min(neg_score, 1.0)
            market_score = min(market_score, 1.0)

            embedding = torch.zeros(news_embedding_dim).to(self.device)
            embedding[0] = length
            embedding[1] = word_count
            embedding[2] = unique_ratio
            embedding[3] = pos_score
            embedding[4] = neg_score
            embedding[5] = market_score
            embedding[6] = pos_score - neg_score
            embedding[7] = (pos_score + neg_score) / 2
            embedding[8] = market_score
            embedding[9] = length * word_count

            embedding[10:20] = torch.randn(10).to(self.device) * noise_scale

            embeddings[i] = embedding

        self.news_encoder.eval()
        with torch.no_grad():
            news_features = self.news_encoder(embeddings)

        return news_features

    def update_market_sentiment(self, sentiment_score: float):
        """Обновление рыночного настроения"""
        norm = self.normalization  # <--- Добавлено!

        alpha = norm.get('sentiment_smoothing_alpha', 0.1)
        self.market_sentiment = (1 - alpha) * self.market_sentiment + alpha * sentiment_score
        self.sentiment_history.append(self.market_sentiment)

        min_history = norm.get('min_history_for_volatility', 10)
        if len(self.sentiment_history) > min_history:
            recent = list(self.sentiment_history)[-min_history:]
            volatility = np.std(recent)
            base_vol = norm.get('base_volatility', 0.7)
            vol_scale = norm.get('volatility_scaling_factor', 3.0)
            self.volatility_index = base_vol + volatility * vol_scale

    def record_strategy_outcome(self, strategy_name: str, action: str,
                                pnl: float, hold_time: float):
        """Запись результата стратегии"""
        perf = self.strategy_performance[strategy_name]
        perf['total_trades'] += 1
        perf['total_pnl'] += pnl

        if pnl > 0:
            perf['profitable_trades'] += 1

        perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
        perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']

    def record_trade_outcome(self, ticker: str, action: str, entry_price: float,
                             exit_price: float, hold_time: float, news_sentiment: float,
                             market_conditions: Dict, strategy: str = None,
                             market_sentiment: float = 0.0) -> Tuple[float, float]:
        """Запись результата сделки"""

        # ✅ ИНИЦИАЛИЗАЦИЯ ВСЕХ ПЕРЕМЕННЫХ В НАЧАЛЕ
        pnl = 0.0
        price_change = 0.0
        norm = self.normalization

        if entry_price > 0 and exit_price > 0:
            price_change = (exit_price - entry_price) / entry_price
            if action == 'SELL':
                # Приоритет: реальный PnL из market_conditions (в рублях)
                pnl = market_conditions.get('pnl', price_change)
            else:
                pnl = 0.0
        else:
            pnl = 0.0
            price_change = 0.0

        stats = self.ticker_stats[ticker]
        stats['total_trades'] += 1

        if action == 'SELL':
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['profitable_trades'] += 1
            if stats['total_trades'] == 1:
                stats['avg_hold_time'] = hold_time
            else:
                stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['total_trades'] - 1) + hold_time) / \
                                         stats['total_trades']

        if stats['total_trades'] > 0:
            stats['success_rate'] = stats['profitable_trades'] / stats['total_trades']

        if strategy and action == 'SELL':
            self.record_strategy_outcome(strategy, action, pnl, hold_time)

        loss_threshold = norm.get('loss_threshold', -0.03)
        if pnl < loss_threshold and action == 'SELL':
            error_data = self.error_memory[ticker]
            error_data['failure_count'] += 1
            # ✅ ИСПРАВЛЕНО: используем time.time() вместо datetime.now().isoformat()
            error_data['last_failure'] = time.time()

        # Расчет награды
        reward = pnl

        if action == 'SELL':
            profit_threshold = norm.get('profit_threshold', 0.05)
            if pnl > profit_threshold:
                reward += self.rl_config.get('good_profit_bonus', 1.0)
            elif pnl < loss_threshold:
                reward -= self.rl_config.get('big_loss_penalty', 1.5)

        return reward, price_change

    def save_model(self):
        """Сохранение модели"""
        try:
            torch.save({
                'news_encoder': self.news_encoder.state_dict(),
                'policy_net': self.policy_net.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
            }, os.path.join(self.model_dir, 'model_weights.pth'))

            state = {
                'error_memory': dict(self.error_memory),
                'ticker_stats': dict(self.ticker_stats),
                'market_sentiment': self.market_sentiment,
                'volatility_index': self.volatility_index,
                'strategy_performance': dict(self.strategy_performance),
                'strategies': self.strategies,
                'save_time': datetime.now().isoformat()
            }

            with open(os.path.join(self.model_dir, 'model_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)

            print(f"[TraderModel] Модель сохранена")

        except Exception as e:
            print(f"[TraderModel] Ошибка сохранения: {e}")

    def load_model(self):
        """Загрузка модели"""
        weights_path = os.path.join(self.model_dir, 'model_weights.pth')
        state_path = os.path.join(self.model_dir, 'model_state.json')

        if os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.news_encoder.load_state_dict(checkpoint['news_encoder'])
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                if 'policy_optimizer' in checkpoint:
                    self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                print(f"[TraderModel] ✓ Загружены веса")
            except Exception as e:
                print(f"[TraderModel] Ошибка загрузки весов: {e}")

        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                self.error_memory.clear()
                for ticker, data in state.get('error_memory', {}).items():
                    self.error_memory[ticker] = data

                self.ticker_stats.clear()
                for ticker, stats in state.get('ticker_stats', {}).items():
                    self.ticker_stats[ticker] = stats

                self.market_sentiment = state.get('market_sentiment', 0.0)
                self.volatility_index = state.get('volatility_index', 1.0)

                if 'strategies' in state:
                    self.strategies = state['strategies']

                if 'strategy_performance' in state:
                    self.strategy_performance.clear()
                    for name, perf in state['strategy_performance'].items():
                        self.strategy_performance[name] = perf

                print(f"[TraderModel] ✓ Загружено состояние")

            except Exception as e:
                print(f"[TraderModel] Ошибка загрузки состояния: {e}")

    def save_memory(self):
        """Сохранение памяти"""
        if not self.memory_config['enable_autosave'] or len(self.memory) == 0:
            return

        try:
            import pickle
            import gzip

            max_to_save = min(self.memory_config['max_memory_to_save'], len(self.memory))
            memory_to_save = list(self.memory)[-max_to_save:]

            file_path = self.memory_config['memory_file']
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if self.memory_config.get('compression', True):
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(memory_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(memory_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[TraderModel] Память сохранена: {len(memory_to_save)} опытов")

        except Exception as e:
            print(f"[TraderModel] Ошибка сохранения памяти: {e}")

    def load_memory(self):
        """Загрузка памяти"""
        file_path = self.memory_config['memory_file']

        if not os.path.exists(file_path):
            return

        try:
            import pickle
            import gzip

            if self.memory_config.get('compression', True):
                with gzip.open(file_path, 'rb') as f:
                    loaded_memory = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    loaded_memory = pickle.load(f)

            self.memory.clear()
            self.memory.extend(loaded_memory)

            if hasattr(self, 'prioritized_buffer'):
                old_alpha = getattr(self.prioritized_buffer, 'alpha',
                                    self.prioritized_buffer_config.get('alpha', 0.6))
                old_beta = getattr(self.prioritized_buffer, 'beta',
                                   self.prioritized_buffer_config.get('beta', 0.4))
                old_beta_inc = getattr(self.prioritized_buffer, 'beta_increment',
                                       self.prioritized_buffer_config.get('beta_increment', 0.001))
                self.prioritized_buffer = PrioritizedReplayBuffer(
                    max_size=self.memory.maxlen,
                    alpha=old_alpha,
                    beta=old_beta,
                    beta_increment=old_beta_inc
                )
                for exp in loaded_memory:
                    self.prioritized_buffer.add(exp)

            print(f"[TraderModel] Загружено {len(loaded_memory)} опытов")

        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки памяти: {e}")


# Глобальный экземпляр
trader_model_instance = AdvancedTraderModel()
NEWS_ENCODED_DIM = trader_model_instance.news_encoded_dim
BASE_STATE_DIM = trader_model_instance.base_state_dim
TOTAL_STATE_DIM = trader_model_instance.total_state_dim