"""
Модернизированная модель трейдера с улучшенной архитектурой
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class NewsEncoder(nn.Module):
    """Улучшенный энкодер новостей с поддержкой разных размерностей"""

    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 128),
            nn.Tanh()
        )

        # Адаптивный слой для разных размерностей
        self.adaptor = nn.Linear(768, input_dim) if input_dim != 768 else nn.Identity()

    def forward(self, x):
        x = self.adaptor(x)
        return self.encoder(x)


class TradingPolicyNetwork(nn.Module):
    """Улучшенная политика с вниманием к новостям"""

    def __init__(self, state_dim=150, action_dim=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Основная сеть для состояния
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Сеть для действий
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )

        # Сеть для оценки состояния
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Механизм внимания для новостей
        self.news_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)

    def forward(self, state, news_features=None):
        # Обработка состояния
        state_features = self.state_net(state)

        # Если есть новости, применяем внимание
        if news_features is not None and news_features.shape[1] == 128:
            # Подготовка для multihead attention
            news_features = news_features.unsqueeze(0)  # [1, batch, features]
            state_features_expanded = state_features.unsqueeze(0)  # [1, batch, features]

            # Применяем внимание (новости как key/value, состояние как query)
            attended, _ = self.news_attention(
                state_features_expanded,
                news_features,
                news_features
            )

            # Объединяем с оригинальными признаками
            state_features = state_features + attended.squeeze(0)

        # Генерация действий и оценки
        action_probs = self.action_net(state_features)
        state_value = self.value_net(state_features)

        return action_probs, state_value


class AdvancedTraderModel:
    """Продвинутая модель трейдера с улучшенной архитектурой"""

    def __init__(self,
                 model_dir: str = "models/saved_trader",
                 learning_rate: float = 0.0005,  # Уменьшенный LR для стабильности
                 gamma: float = 0.95,
                 memory_size: int = 5000):  # Уменьшенная память для эффективности

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Инициализация сетей с правильными размерностями
        self.news_encoder = NewsEncoder(input_dim=768, hidden_dim=256)
        self.news_encoder.to(self.device)

        self.policy_net = TradingPolicyNetwork(state_dim=150, action_dim=3)
        self.policy_net.to(self.device)

        # Загрузка BERT для русского языка
        self.bert_model, self.bert_tokenizer = self._load_bert_model()

        # Оптимизаторы
        self.news_optimizer = optim.AdamW(self.news_encoder.parameters(),
                                          lr=learning_rate,
                                          weight_decay=0.01)
        self.policy_optimizer = optim.AdamW(self.policy_net.parameters(),
                                            lr=learning_rate,
                                            weight_decay=0.01)

        # Память и состояние
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma

        # Статистика
        self.error_memory = defaultdict(lambda: {
            'failed_trades': [],
            'avg_loss': 0.0,
            'last_failure': None,
            'failure_count': 0,
            'success_rate': 0.5,
            'total_trades': 0
        })

        self.ticker_stats = defaultdict(lambda: {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0,
            'success_rate': 0.5,
            'last_trade': None
        })

        # Рыночное состояние
        self.market_sentiment = 0.0
        self.sentiment_history = deque(maxlen=200)
        self.volatility_index = 1.0

        # Загрузка сохраненной модели
        self.load_model()

        print(f"[TraderModel] Инициализирована на {self.device}")
        print(f"[TraderModel] Статистика: {len(self.error_memory)} тикеров, "
              f"{len(self.memory)} опытов, sentiment={self.market_sentiment:.3f}")

    def _load_bert_model(self):
        """Загрузка BERT модели с обработкой ошибок"""
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
            print("[TraderModel] ⚠ Используется упрощенный анализ новостей")
            return None, None

    def save_model(self):
        """Улучшенное сохранение модели"""
        try:
            # Сохраняем веса
            torch.save({
                'news_encoder': self.news_encoder.state_dict(),
                'policy_net': self.policy_net.state_dict(),
                'news_optimizer': self.news_optimizer.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'model_config': {
                    'bert_model': 'cointegrated/rubert-tiny2',
                    'news_input_dim': 768,
                    'state_dim': 150
                }
            }, os.path.join(self.model_dir, 'model_weights.pth'))

            # Сохраняем состояние
            state = {
                'error_memory': dict(self.error_memory),
                'ticker_stats': dict(self.ticker_stats),
                'market_sentiment': self.market_sentiment,
                'sentiment_history': list(self.sentiment_history),
                'volatility_index': self.volatility_index,
                'memory_size': len(self.memory),
                'total_experiences': sum(len(v['failed_trades']) for v in self.error_memory.values()),
                'save_time': datetime.now().isoformat()
            }

            with open(os.path.join(self.model_dir, 'model_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)

            print(f"[TraderModel] Модель сохранена: {len(self.memory)} опытов, "
                  f"{len(self.error_memory)} тикеров")

        except Exception as e:
            print(f"[TraderModel] Ошибка сохранения: {e}")

    def load_model(self):
        """Улучшенная загрузка модели"""
        weights_path = os.path.join(self.model_dir, 'model_weights.pth')
        state_path = os.path.join(self.model_dir, 'model_state.json')

        # Загрузка весов
        if os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)

                # Загружаем веса
                self.news_encoder.load_state_dict(checkpoint['news_encoder'])
                self.policy_net.load_state_dict(checkpoint['policy_net'])

                # Загружаем оптимизаторы (если есть)
                if 'news_optimizer' in checkpoint:
                    self.news_optimizer.load_state_dict(checkpoint['news_optimizer'])
                if 'policy_optimizer' in checkpoint:
                    self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

                print(f"[TraderModel] ✓ Загружены веса нейросетей")

            except Exception as e:
                print(f"[TraderModel] Ошибка загрузки весов: {e}")

        # Загрузка состояния
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # Восстанавливаем память ошибок
                self.error_memory.clear()
                for ticker, data in state.get('error_memory', {}).items():
                    self.error_memory[ticker] = data

                # Восстанавливаем статистику
                self.ticker_stats.clear()
                for ticker, stats in state.get('ticker_stats', {}).items():
                    self.ticker_stats[ticker] = stats

                # Восстанавливаем рыночное состояние
                self.market_sentiment = state.get('market_sentiment', 0.0)
                self.sentiment_history = deque(
                    state.get('sentiment_history', []),
                    maxlen=200
                )
                self.volatility_index = state.get('volatility_index', 1.0)

                print(f"[TraderModel] ✓ Загружено состояние: {len(self.error_memory)} тикеров, "
                      f"senti={self.market_sentiment:.3f}, vol={self.volatility_index:.2f}")

            except Exception as e:
                print(f"[TraderModel] Ошибка загрузки состояния: {e}")

    def encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Кодирование новостей через BERT с fallback"""
        if not news_texts:
            return torch.zeros(1, 128).to(self.device)

        # Пробуем BERT
        if self.bert_model is not None and self.bert_tokenizer is not None:
            try:
                # Токенизация
                inputs = self.bert_tokenizer(
                    news_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                # Получение эмбеддингов
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

                # Кодирование через энкодер
                self.news_encoder.eval()
                with torch.no_grad():
                    news_features = self.news_encoder(embeddings)

                return news_features

            except Exception as e:
                print(f"[TraderModel] Ошибка BERT кодирования: {e}")

        # Fallback: упрощенное кодирование
        return self.simple_encode_news(news_texts)

    def simple_encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Упрощенное кодирование новостей"""
        batch_size = len(news_texts)
        embeddings = torch.zeros(batch_size, 768).to(self.device)

        sentiment_dict = {
            'positive': ['рост', 'прибыль', 'увеличение', 'дивиденд', 'выше', 'улучшение', 'рекомендуют', 'покупать'],
            'negative': ['падение', 'убыток', 'снижение', 'проблемы', 'ниже', 'сокращение', 'продавать', 'снижают'],
            'market': ['рынок', 'акци', 'бирж', 'фондов', 'инвест', 'торг', 'ликвид', 'волатиль']
        }

        for i, text in enumerate(news_texts):
            text_lower = text.lower()

            # Базовые признаки
            length = min(len(text) / 1000, 1.0)
            words = text_lower.split()
            word_count = min(len(words) / 200, 1.0)
            unique_ratio = len(set(words)) / max(len(words), 1)

            # Сентимент признаки
            pos_score = sum(1 for w in sentiment_dict['positive'] if w in text_lower) / 10
            neg_score = sum(1 for w in sentiment_dict['negative'] if w in text_lower) / 10
            market_score = sum(1 for w in sentiment_dict['market'] if w in text_lower) / 10

            # Нормализация
            pos_score = min(pos_score, 1.0)
            neg_score = min(neg_score, 1.0)
            market_score = min(market_score, 1.0)

            # Создание эмбеддинга
            embedding = torch.zeros(768).to(self.device)

            # Заполняем первые 10 признаков
            embedding[0] = length
            embedding[1] = word_count
            embedding[2] = unique_ratio
            embedding[3] = pos_score
            embedding[4] = neg_score
            embedding[5] = market_score
            embedding[6] = pos_score - neg_score  # чистый сентимент
            embedding[7] = (pos_score + neg_score) / 2  # интенсивность
            embedding[8] = market_score
            embedding[9] = length * word_count  # сложность текста

            # Добавляем немного случайности для разнообразия
            embedding[10:20] = torch.randn(10).to(self.device) * 0.1

            embeddings[i] = embedding

        # Пропускаем через энкодер
        self.news_encoder.eval()
        with torch.no_grad():
            news_features = self.news_encoder(embeddings)

        return news_features

    def calculate_risk_score(self, ticker: str, price: float, sentiment: float) -> float:
        """Расчет риск-скора с учетом волатильности"""
        error_data = self.error_memory[ticker]
        stats = self.ticker_stats[ticker]

        # Базовый риск
        if error_data['failure_count'] == 0:
            base_risk = 0.4  # Ниже для новых тикеров
        else:
            failure_penalty = min(error_data['failure_count'] * 0.08, 0.8)
            loss_penalty = min(abs(error_data['avg_loss']) * 1.5, 0.6)
            success_bonus = max(stats['success_rate'] - 0.5, 0) * 0.3

            base_risk = 0.4 + failure_penalty + loss_penalty - success_bonus

        # Корректировка на сентимент
        sentiment_factor = 1.2 - abs(sentiment)  # Нейтральный = больше риска

        # Корректировка на волатильность рынка
        volatility_factor = 0.8 + (self.volatility_index * 0.4)

        final_risk = base_risk * sentiment_factor * volatility_factor

        # Ограничение диапазона
        return max(0.1, min(0.85, final_risk))

    def build_state_vector(self,
                           ticker: str,
                           price: float,
                           momentum: float,
                           sentiment: float,
                           news_features: torch.Tensor,
                           market_data: Dict) -> torch.Tensor:
        """Построение вектора состояния"""

        # Признаки из новостей
        if news_features.numel() > 0 and news_features.shape[1] == 128:
            news_vec = news_features.mean(dim=0).cpu().numpy()
        else:
            news_vec = np.zeros(128)

        # Статистика тикера
        stats = self.ticker_stats[ticker]
        success_rate = stats['success_rate']
        total_trades = stats['total_trades']
        avg_hold_time = stats['avg_hold_time']

        # Расчет риска
        risk_score = self.calculate_risk_score(ticker, price, sentiment)

        # Базовые признаки (22 шт)
        features = [
            # Ценовые признаки
            price / 10000.0,  # Нормализованная цена
            momentum * 20.0,  # Моментум
            sentiment,  # Сентимент

            # Риск и статистика
            risk_score,
            success_rate,
            min(total_trades / 50.0, 2.0),  # Опыт (нормализованный)
            min(avg_hold_time / 24.0, 7.0),  # Время удержания (в днях)

            # Рыночные признаки
            self.market_sentiment,
            self.volatility_index,

            # Данные рынка
            market_data.get('volume', 0) / 1e7,  # Объем (десятки млн)
            market_data.get('spread', 0.01) * 100,  # Спред (%)
            market_data.get('liquidity', 0.5),  # Ликвидность
            market_data.get('rsi', 50) / 100.0,  # RSI
            market_data.get('volatility', 0.1) * 10,  # Волатильность

            # Технические признаки
            market_data.get('sma_10_ratio', 1.0),  # Отношение к SMA10
            market_data.get('sma_20_ratio', 1.0),  # Отношение к SMA20
            market_data.get('bb_position', 0.5),  # Позиция в BB
            market_data.get('volume_ratio', 1.0),  # Отношение объема

            # Временные признаки
            datetime.now().hour / 24.0,  # Время суток
            datetime.now().weekday() / 7.0,  # День недели

            # Дополнительные
            market_data.get('market_cap', 0) / 1e12,  # Капитализация (триллионы)
            market_data.get('pe_ratio', 15) / 100.0,  # P/E
        ]

        # Новостные признаки (128)
        features.extend(news_vec.tolist())

        # ИТОГО: 22 + 128 = 150 признаков

        # Преобразование в тензор
        state_vector = torch.FloatTensor(features).to(self.device)

        # Проверка размерности
        if state_vector.shape[0] != 150:
            # Автоматическая корректировка
            if state_vector.shape[0] < 150:
                padding = torch.zeros(150 - state_vector.shape[0]).to(self.device)
                state_vector = torch.cat([state_vector, padding])
            else:
                state_vector = state_vector[:150]

        return state_vector

    def choose_action(self,
                      state: torch.Tensor,
                      ticker: str,
                      current_price: float) -> Tuple[int, float, float]:
        """Выбор действия с улучшенной логикой"""
        self.policy_net.eval()

        with torch.no_grad():
            # Получаем вероятности действий
            action_probs, state_value = self.policy_net(state.unsqueeze(0))

        probs = action_probs.cpu().numpy().flatten()

        # Корректировка на основе истории
        error_data = self.error_memory[ticker]
        stats = self.ticker_stats[ticker]

        if error_data['failure_count'] > 1:
            # Корректировка для проблемных тикеров
            failure_factor = min(error_data['failure_count'] * 0.2, 0.6)

            # Снижаем вероятность покупки, повышаем продажи
            probs[0] *= (1.0 - failure_factor)  # BUY
            probs[2] *= (1.0 + failure_factor)  # SELL

            # Нормализация
            probs = probs / probs.sum()

        # Корректировка на основе успешности
        if stats['total_trades'] > 5:
            success_factor = max(stats['success_rate'] - 0.5, 0) * 0.4

            if stats['success_rate'] > 0.6:
                # Успешные тикеры - больше покупок
                probs[0] *= (1.0 + success_factor)  # BUY
            elif stats['success_rate'] < 0.4:
                # Неуспешные тикеры - больше продаж
                probs[2] *= (1.0 + 0.3)  # SELL

            probs = probs / probs.sum()

        # Выбор действия (с эксплорацией на ранних этапах)
        exploration_rate = max(0.1, 0.3 - (len(self.memory) / 10000) * 0.2)

        if np.random.random() < exploration_rate and len(self.memory) < 3000:
            # Случайное действие для исследования
            action = np.random.choice(len(probs))
        else:
            # Жадное действие
            action = np.argmax(probs)

        confidence = probs[action]

        return action, confidence, state_value.item()

    def remember_experience(self,
                            state: torch.Tensor,
                            action: int,
                            reward: float,
                            next_state: torch.Tensor,
                            done: bool,
                            news_features: Optional[torch.Tensor] = None):
        """Сохранение опыта с новостями"""
        self.memory.append({
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            'news_features': news_features.cpu() if news_features is not None else None,
            'timestamp': datetime.now().isoformat()
        })

    def learn_from_experience(self, batch_size: int = 32):
        """Обучение на опыте с улучшенной стабильностью"""
        if len(self.memory) < batch_size * 2:
            return None

        try:
            # Выборка с приоритетом (простейшая реализация)
            recent_size = min(len(self.memory) // 4, batch_size // 2)
            recent_indices = list(range(len(self.memory) - recent_size, len(self.memory)))
            random_indices = np.random.choice(len(self.memory) - recent_size,
                                              batch_size - recent_size,
                                              replace=False)
            indices = list(random_indices) + recent_indices

            batch = [self.memory[i] for i in indices]

            # Подготовка данных
            states = torch.stack([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

            # Новостные признаки (если есть)
            news_features_list = []
            for exp in batch:
                if exp['news_features'] is not None:
                    news_features_list.append(exp['news_features'])

            if news_features_list:
                news_features = torch.stack(news_features_list).to(self.device)
            else:
                news_features = None

            # Переключаем в режим обучения
            self.policy_net.train()

            # Текущие оценки
            if news_features is not None:
                current_probs, current_values = self.policy_net(states, news_features)
            else:
                current_probs, current_values = self.policy_net(states)

            # Следующие оценки (без градиентов)
            with torch.no_grad():
                if news_features is not None:
                    _, next_values = self.policy_net(next_states, news_features)
                else:
                    _, next_values = self.policy_net(next_states)

            # Целевые значения
            target_values = rewards + (1 - dones) * self.gamma * next_values

            # Value loss
            value_loss = nn.SmoothL1Loss()(current_values, target_values.detach())

            # Policy loss
            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)

            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            # Entropy regularization
            entropy = dist.entropy().mean()
            entropy_bonus = 0.01 * entropy

            # Общий loss
            total_loss = value_loss + policy_loss - entropy_bonus

            # Оптимизация
            self.policy_optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)

            self.policy_optimizer.step()

            # Периодическое обучение энкодера новостей
            if len(self.memory) % 100 == 0 and news_features is not None:
                self._train_news_encoder(batch)

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка обучения: {e}")
            return None

    def _train_news_encoder(self, batch):
        """Обучение энкодера новостей"""
        try:
            # Простая задача реконструкции
            news_features = torch.stack([exp['news_features'] for exp in batch
                                         if exp['news_features'] is not None]).to(self.device)

            if len(news_features) < 4:
                return

            self.news_encoder.train()

            # Прямой проход
            encoded = self.news_encoder(news_features)

            # Декодер (простой линейный слой)
            decoder = nn.Linear(128, news_features.shape[1]).to(self.device)

            # Реконструкция
            reconstructed = decoder(encoded)

            # Loss реконструкции
            recon_loss = nn.MSELoss()(reconstructed, news_features)

            # Оптимизация
            self.news_optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.news_encoder.parameters(), 1.0)
            self.news_optimizer.step()

            self.news_encoder.eval()

        except Exception as e:
            print(f"[TraderModel] Ошибка обучения энкодера: {e}")

    def update_market_sentiment(self, sentiment_score: float):
        """Обновление рыночного настроения"""
        # Экспоненциальное сглаживание
        alpha = 0.1
        self.market_sentiment = (1 - alpha) * self.market_sentiment + alpha * sentiment_score

        # Сохранение в историю
        self.sentiment_history.append(self.market_sentiment)

        # Обновление индекса волатильности на основе колебаний сентимента
        if len(self.sentiment_history) > 10:
            recent_sentiments = list(self.sentiment_history)[-10:]
            volatility = np.std(recent_sentiments)
            self.volatility_index = 0.7 + volatility * 3.0  # Масштабирование

    def record_trade_outcome(self,
                             ticker: str,
                             action: str,
                             entry_price: float,
                             exit_price: float,
                             hold_time: float,
                             news_sentiment: float,
                             market_conditions: Dict) -> Tuple[float, float]:
        """Запись результата сделки"""

        # Расчет PnL
        if action == 'BUY':
            # Для покупки PnL рассчитывается при продаже
            pnl = 0.0
        elif action == 'SELL':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = 0.0

        # Обновление статистики тикера
        stats = self.ticker_stats[ticker]
        stats['total_trades'] += 1

        if action == 'SELL':
            stats['total_pnl'] += pnl

            if pnl > 0:
                stats['profitable_trades'] += 1

            # Обновление среднего времени удержания
            if stats['total_trades'] == 1:
                stats['avg_hold_time'] = hold_time
            else:
                stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['total_trades'] - 1) + hold_time) / stats[
                    'total_trades']

        stats['success_rate'] = stats['profitable_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0.5
        stats['last_trade'] = datetime.now().isoformat()

        # Запись ошибок (значительные убытки)
        if pnl < -0.02:  # Убыток более 2%
            error_data = self.error_memory[ticker]

            trade_record = {
                'date': datetime.now().isoformat(),
                'action': action,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'hold_time': hold_time,
                'sentiment': news_sentiment,
                'market_conditions': market_conditions
            }

            error_data['failed_trades'].append(trade_record)
            error_data['failure_count'] += 1
            error_data['last_failure'] = datetime.now().isoformat()

            # Обновление среднего убытка
            if error_data['failed_trades']:
                losses = [t['pnl'] for t in error_data['failed_trades']]
                error_data['avg_loss'] = sum(losses) / len(losses)

            # Обновление success rate в error memory
            error_data['success_rate'] = stats['success_rate']
            error_data['total_trades'] = stats['total_trades']

            print(f"[TraderModel] Запомнена ошибка: {ticker} {action}, убыток {pnl:.2%}")

        # Награда для обучения
        if action == 'SELL':
            reward = pnl * 20.0  # Масштабирование

            # Бонусы/штрафы
            if hold_time < 0.5 and abs(pnl) < 0.01:  # Слишком быстро
                reward -= 0.5
            elif pnl > 0.05:  # Хорошая прибыль
                reward += 1.0
            elif pnl < -0.03:  # Большой убыток
                reward -= 1.5
            elif pnl > 0:  # Небольшая прибыль
                reward += 0.2
        else:
            reward = 0.0

        return reward, pnl

    def rank_candidates(self,
                        prices: Dict[str, float],
                        securities: Dict[str, Dict],
                        ticker_sentiment: Dict[str, float],
                        news_by_ticker: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """Ранжирование кандидатов с улучшенной логикой"""

        candidates = []

        for ticker, price in prices.items():
            if ticker not in securities:
                continue

            market_data = securities[ticker]
            momentum = market_data.get('momentum', 0.0)
            sentiment = ticker_sentiment.get(ticker, self.market_sentiment)

            # Новости
            ticker_news = news_by_ticker.get(ticker, [])
            if ticker_news:
                news_features = self.encode_news(ticker_news[:3])
            else:
                news_features = torch.zeros(1, 128).to(self.device)

            # Построение состояния
            state = self.build_state_vector(
                ticker=ticker,
                price=price,
                momentum=momentum,
                sentiment=sentiment,
                news_features=news_features,
                market_data=market_data
            )

            # Выбор действия
            try:
                action, confidence, _ = self.choose_action(state, ticker, price)
            except Exception as e:
                print(f"[TraderModel] Ошибка choose_action для {ticker}: {e}")
                action, confidence = 1, 0.5  # HOLD по умолчанию

            # Оценка кандидата
            stats = self.ticker_stats[ticker]
            error_data = self.error_memory[ticker]

            if action == 0:  # BUY
                # Факторы для покупки
                success_bonus = max(stats['success_rate'] - 0.5, 0) * 0.4
                risk_score = self.calculate_risk_score(ticker, price, sentiment)
                risk_penalty = risk_score * 0.3

                # Награда за низкую волатильность
                volatility_penalty = market_data.get('volatility', 0.1) * 0.5

                score = (confidence * 0.5 +
                         sentiment * 0.3 +
                         success_bonus -
                         risk_penalty -
                         volatility_penalty +
                         self.market_sentiment * 0.2 +
                         momentum * 0.2)

            elif action == 2:  # SELL
                # Факторы для продажи
                failure_penalty = min(error_data['failure_count'] * 0.15, 0.6)

                # Если позиция в убытке - выше приоритет продажи
                current_return = stats.get('current_return', 0.0)
                loss_bonus = max(-current_return * 0.3, 0) if current_return < 0 else 0

                score = (confidence * 0.4 -
                         sentiment * 0.3 +
                         failure_penalty +
                         loss_bonus +
                         (-momentum * 0.3))  # Отрицательный моментум для продажи

            else:  # HOLD
                score = 0.0

            candidates.append((ticker, score))

        # Сортировка
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"[TraderModel] Отранжировано {len(candidates)} кандидатов")

        return candidates[:30]  # Топ-30

    def get_worst_position(self,
                           positions: Dict[str, Dict],
                           prices: Dict[str, float]) -> Tuple[Optional[str], int]:
        """Определение худшей позиции"""
        if not positions:
            return None, 0

        worst_score = float('inf')
        worst_ticker = None

        for ticker, pos_data in positions.items():
            if ticker not in prices:
                continue

            current_price = prices[ticker]
            avg_price = pos_data['avg_price']

            if avg_price > 0:
                pnl_ratio = (current_price - avg_price) / avg_price
            else:
                pnl_ratio = 0.0

            # История ошибок
            error_data = self.error_memory[ticker]
            failure_penalty = error_data['failure_count'] * 0.2

            # Время удержания
            hold_time_days = pos_data.get('hold_time_days', 0)
            hold_penalty = min(hold_time_days / 60.0, 1.0) * 0.3  # После 60 дней макс штраф

            # Общий score (чем ниже, тем хуже)
            score = pnl_ratio - failure_penalty - hold_penalty

            if score < worst_score:
                worst_score = score
                worst_ticker = ticker

        if worst_ticker:
            pos = positions[worst_ticker]
            # Продаем от 25% до 50% позиции в зависимости от "плохости"
            severity = min((0.5 - worst_score) / 0.5, 1.0)  # Нормализация
            sell_percentage = 0.25 + severity * 0.25  # 25-50%

            qty = int(pos['qty'] * sell_percentage)
            qty = max(qty, 1)  # Минимум 1 акция

            return worst_ticker, qty

        return None, 0

    def periodic_learning(self):
        """Периодическое обучение"""
        if len(self.memory) > 100:
            loss = self.learn_from_experience(batch_size=32)

            # Сохранение каждые 50 шагов обучения
            if not hasattr(self, '_learn_steps'):
                self._learn_steps = 0

            self._learn_steps += 1

            if self._learn_steps % 50 == 0:
                self.save_model()
                print(f"[TraderModel] Автосохранение после {self._learn_steps} шагов обучения")

            return loss

        return None

    def get_model_stats(self) -> Dict:
        """Получение статистики модели"""
        return {
            'memory_size': len(self.memory),
            'ticker_stats_count': len(self.ticker_stats),
            'error_memory_count': len(self.error_memory),
            'market_sentiment': self.market_sentiment,
            'volatility_index': self.volatility_index,
            'device': str(self.device),
            'last_learn_steps': getattr(self, '_learn_steps', 0)
        }


# Глобальный экземпляр
trader_model_instance = AdvancedTraderModel()