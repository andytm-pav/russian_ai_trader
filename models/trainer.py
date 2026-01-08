"""
Модуль для фонового обучения модели трейдера
"""

import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from utils.logger import setup_logger
from models.trader_model import trader_model_instance

logger = setup_logger("TRAINER")


class ModelTrainer:
    """Фоновое обучение модели трейдера"""

    def __init__(self,
                 model=None,
                 training_interval: int = 300,  # секунд между обучениями
                 batch_size: int = 32,
                 save_interval: int = 10):  # сохранять каждые N обучений

        self.model = model or trader_model_instance
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.save_interval = save_interval

        self.training_enabled = True
        self.training_thread = None
        self.training_stats = {
            'total_training_sessions': 0,
            'avg_loss': 0.0,
            'last_training': None,
            'loss_history': []
        }

        logger.info(f"Инициализирован ModelTrainer (интервал: {training_interval}с)")

    def start_background_training(self):
        """Запуск фонового обучения"""
        if self.training_thread and self.training_thread.is_alive():
            logger.warning("Обучение уже запущено")
            return

        self.training_enabled = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
            name="ModelTrainer"
        )
        self.training_thread.start()
        logger.info("Фоновое обучение запущено")

    def stop_background_training(self):
        """Остановка фонового обучения"""
        self.training_enabled = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
            logger.info("Фоновое обучение остановлено")

    def _training_loop(self):
        """Цикл фонового обучения"""
        training_counter = 0

        while self.training_enabled:
            try:
                # Ждем интервал
                time.sleep(self.training_interval)

                # Проверяем, достаточно ли данных для обучения
                if len(self.model.memory) < self.batch_size * 2:
                    logger.debug(f"Недостаточно данных для обучения: {len(self.model.memory)} < {self.batch_size * 2}")
                    continue

                # Обучение
                loss = self.model.learn_from_experience(batch_size=self.batch_size)

                if loss is not None:
                    training_counter += 1

                    # Обновление статистики
                    self.training_stats['total_training_sessions'] += 1
                    self.training_stats['last_training'] = datetime.now().isoformat()
                    self.training_stats['loss_history'].append(loss)

                    # Сохраняем только последние 100 значений
                    if len(self.training_stats['loss_history']) > 100:
                        self.training_stats['loss_history'] = self.training_stats['loss_history'][-100:]

                    # Расчет среднего лосса
                    self.training_stats['avg_loss'] = np.mean(self.training_stats['loss_history'][-10:])  # последние 10

                    # Логирование
                    logger.info(f"Обучение #{training_counter}: Loss = {loss:.6f}, "
                                f"Память = {len(self.model.memory)}, "
                                f"Ср. Loss = {self.training_stats['avg_loss']:.6f}")

                    # Периодическое сохранение
                    if training_counter % self.save_interval == 0:
                        self.model.save_model()
                        logger.info(f"Модель сохранена после {training_counter} обучений")

                # Очистка старой памяти (если слишком много)
                self._cleanup_old_memory()

            except Exception as e:
                logger.error(f"Ошибка в цикле обучения: {e}")
                time.sleep(60)  # Подождать перед следующей попыткой

    def _cleanup_old_memory(self):
        """Очистка старой памяти для эффективности"""
        max_memory_size = 10000

        if len(self.model.memory) > max_memory_size:
            # Удаляем самые старые записи
            remove_count = len(self.model.memory) - max_memory_size
            for _ in range(remove_count):
                self.model.memory.popleft()

            logger.debug(f"Очищена память: удалено {remove_count} старых записей")

    def train_on_historical_data(self,
                                 historical_file: str = "data/historical_trades.json"):
        """Обучение на исторических данных"""
        try:
            logger.info(f"Начинаю обучение на исторических данных из {historical_file}")

            # Загрузка исторических данных
            with open(historical_file, 'r', encoding='utf-8') as f:
                historical_data = json.load(f)

            trades = historical_data.get('trades', [])

            if not trades:
                logger.warning("Исторические данные пусты")
                return

            logger.info(f"Загружено {len(trades)} исторических сделок")

            # Обучение на каждой сделке
            for i, trade in enumerate(trades, 1):
                try:
                    # Создание фиктивного состояния для обучения
                    ticker = trade.get('ticker', 'UNKNOWN')
                    action = trade.get('action', 'HOLD')

                    # Создание состояния
                    state = torch.randn(150).to(self.model.device) * 0.1

                    # Определение награды на основе PnL
                    pnl = trade.get('pnl', 0.0)
                    reward = pnl * 10.0  # Масштабирование

                    # Следующее состояние (похожее)
                    next_state = state + torch.randn(150).to(self.model.device) * 0.01

                    # Действие (0=BUY, 1=HOLD, 2=SELL)
                    if action == 'BUY':
                        action_idx = 0
                    elif action == 'SELL':
                        action_idx = 2
                    else:
                        action_idx = 1

                    done = True  # Сделка завершена

                    # Сохраняем в память
                    self.model.remember_experience(
                        state=state,
                        action=action_idx,
                        reward=reward,
                        next_state=next_state,
                        done=done
                    )

                    # Периодическое обучение
                    if i % 10 == 0 and len(self.model.memory) >= self.batch_size:
                        loss = self.model.learn_from_experience(batch_size=min(self.batch_size, len(self.model.memory)))
                        if loss is not None:
                            logger.debug(f"Обучение на историч. данных #{i}: Loss = {loss:.6f}")

                except Exception as e:
                    logger.error(f"Ошибка обработки сделки #{i}: {e}")

            # Финальное обучение
            if len(self.model.memory) >= self.batch_size:
                final_loss = self.model.learn_from_experience(batch_size=self.batch_size)
                logger.info(f"Финальное обучение на историч. данных: Loss = {final_loss:.6f}")

            # Сохранение модели
            self.model.save_model()
            logger.info("Обучение на исторических данных завершено")

        except FileNotFoundError:
            logger.warning(f"Файл исторических данных не найден: {historical_file}")
        except Exception as e:
            logger.error(f"Ошибка обучения на исторических данных: {e}")

    def generate_training_report(self) -> Dict:
        """Генерация отчета по обучению"""
        model_stats = self.model.get_model_stats()

        report = {
            'training_stats': self.training_stats,
            'model_stats': model_stats,
            'training_enabled': self.training_enabled,
            'memory_usage': {
                'memory_size': len(self.model.memory),
                'memory_max_size': self.model.memory.maxlen if hasattr(self.model.memory, 'maxlen') else 0,
                'memory_percent': (len(self.model.memory) / self.model.memory.maxlen * 100)
                if hasattr(self.model.memory, 'maxlen') and self.model.memory.maxlen > 0 else 0
            },
            'training_config': {
                'training_interval': self.training_interval,
                'batch_size': self.batch_size,
                'save_interval': self.save_interval
            },
            'report_time': datetime.now().isoformat()
        }

        # Добавляем историю лосса
        if self.training_stats['loss_history']:
            report['loss_trend'] = {
                'min_loss': min(self.training_stats['loss_history']),
                'max_loss': max(self.training_stats['loss_history']),
                'last_10_avg': np.mean(self.training_stats['loss_history'][-10:]) if len(
                    self.training_stats['loss_history']) >= 10 else None,
                'improvement': (self.training_stats['loss_history'][0] - self.training_stats['loss_history'][-1])
                if len(self.training_stats['loss_history']) >= 2 else 0
            }

        return report

    def export_training_data(self, export_path: str = "data/training_export.json"):
        """Экспорт данных обучения для анализа"""
        try:
            # Собираем данные
            export_data = {
                'training_stats': self.training_stats,
                'model_memory_sample': [],
                'error_memory_summary': {},
                'ticker_stats_summary': {},
                'export_time': datetime.now().isoformat()
            }

            # Примеры из памяти
            sample_size = min(50, len(self.model.memory))
            if sample_size > 0:
                indices = np.random.choice(len(self.model.memory), sample_size, replace=False)
                for idx in indices:
                    memory_item = list(self.model.memory)[idx]
                    export_data['model_memory_sample'].append({
                        'action': memory_item.get('action'),
                        'reward': float(memory_item.get('reward', 0)),
                        'done': memory_item.get('done', False)
                    })

            # Статистика по ошибкам
            for ticker, data in list(self.model.error_memory.items())[:20]:  # первые 20
                export_data['error_memory_summary'][ticker] = {
                    'failure_count': data.get('failure_count', 0),
                    'avg_loss': data.get('avg_loss', 0.0),
                    'success_rate': data.get('success_rate', 0.5)
                }

            # Статистика по тикерам
            for ticker, stats in list(self.model.ticker_stats.items())[:20]:  # первые 20
                export_data['ticker_stats_summary'][ticker] = {
                    'total_trades': stats.get('total_trades', 0),
                    'success_rate': stats.get('success_rate', 0.5),
                    'avg_hold_time': stats.get('avg_hold_time', 0.0)
                }

            # Сохранение
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Данные обучения экспортированы в {export_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка экспорта данных обучения: {e}")
            return False

    def train_strategies(self):
        """Обучение выбору стратегий"""
        if len(self.model.strategy_memory) < 50:
            return

        # Создаем dataset из стратегий
        strategy_data = []

        for memory in self.model.strategy_memory:
            strategy_data.append({
                'strategy': memory['strategy'],
                'pnl': memory['pnl'],
                'hold_time': memory['hold_time'],
                'action': memory['action']
            })

        # Анализ эффективности стратегий
        df = pd.DataFrame(strategy_data)
        strategy_stats = df.groupby('strategy').agg({
            'pnl': ['mean', 'std', 'count'],
            'hold_time': 'mean'
        })

        logger.info(f"Статистика стратегий:\n{strategy_stats}")


# Глобальный экземпляр
model_trainer_instance = ModelTrainer()