#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆕 v16.4: Дообучение новостной ML-модели сентимента на CPU.

Позволяет пользователю корректировать сентимент новостей через веб-дашборд
и дообучать модель на основе собранных коррекций.

Файлы:
  - data/sentiment_corrections.json — накопленные коррекции
  - data/models/<model_name>_finetuned/ — дообученная модель

Логика:
  1. Пользователь в веб-дашборде нажимает на новость → выбирает POSITIVE/NEGATIVE/NEUTRAL
  2. Коррекция сохраняется в sentiment_corrections.json
  3. Пользователь нажимает "Дообучить модель"
  4. Модуль загружает базовую модель, fine-tunes на коррекциях (CPU), сохраняет
  5. NewsAnalyzer при следующей инициализации загружает дообученную модель
"""
import json
import os
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("SENTIMENT_FT")

# Пути
CORRECTIONS_FILE = Path("data/sentiment_corrections.json")
BASE_MODEL_DIR = Path("data/models")
FINETUNED_MODEL_DIR = BASE_MODEL_DIR / "mxlcw_rubert-tiny2-russian-financial-sentiment_finetuned"
BASE_MODEL_NAME = "mxlcw/rubert-tiny2-russian-financial-sentiment"

# Маппинг лейблов для модели
LABEL_MAP = {
    'POSITIVE': 'positive',
    'NEGATIVE': 'negative',
    'NEUTRAL': 'neutral',
}


def save_correction(news_title: str, news_summary: str, source: str,
                    original_label: str, original_sentiment: float,
                    corrected_label: str, corrected_by: str = "user") -> bool:
    """
    Сохранение коррекции сентимента пользователем.

    Args:
        news_title: заголовок новости
        news_summary: краткое содержание
        source: источник
        original_label: исходный лейбл модели (POSITIVE/NEGATIVE/NEUTRAL)
        original_sentiment: исходный sentiment (-1..1)
        corrected_label: исправленный лейбл
        corrected_by: кто исправил

    Returns:
        True если сохранено
    """
    try:
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Загружаем существующие коррекции
        corrections = []
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
                corrections = json.load(f)

        # Хеш для дедупликации
        text_hash = hashlib.md5(news_title.encode()).hexdigest()

        # Проверяем, нет ли уже коррекции для этой новости
        existing = [c for c in corrections if c.get('text_hash') == text_hash]
        if existing:
            # Обновляем существующую
            existing[0].update({
                'corrected_label': corrected_label,
                'corrected_at': datetime.now().isoformat(),
                'corrected_by': corrected_by,
            })
        else:
            corrections.append({
                'text_hash': text_hash,
                'title': news_title,
                'summary': news_summary,
                'source': source,
                'original_label': original_label,
                'original_sentiment': original_sentiment,
                'corrected_label': corrected_label,
                'corrected_at': datetime.now().isoformat(),
                'corrected_by': corrected_by,
            })

        with open(CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(corrections, f, indent=2, ensure_ascii=False)

        logger.info(f"Коррекция сохранена: '{news_title[:60]}' → {corrected_label} "
                    f"(всего коррекций: {len(corrections)})")
        return True

    except Exception as e:
        logger.error(f"Ошибка сохранения коррекции: {e}")
        return False


def get_corrections() -> List[Dict]:
    """Загрузка всех коррекций."""
    try:
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки коррекций: {e}")
    return []


def get_corrections_count() -> int:
    """Количество коррекций."""
    return len(get_corrections())


def delete_correction(text_hash: str) -> bool:
    """Удаление коррекции по хешу."""
    try:
        corrections = get_corrections()
        new_list = [c for c in corrections if c.get('text_hash') != text_hash]
        with open(CORRECTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(new_list, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Ошибка удаления коррекции: {e}")
        return False


def finetune_model(epochs: int = 3, batch_size: int = 8,
                   learning_rate: float = 2e-5) -> Dict:
    """
    Дообучение модели на собранных коррекциях.

    Args:
        epochs: количество эпох
        batch_size: размер батча (CPU → маленький)
        learning_rate: скорость обучения

    Returns:
        {'success': bool, 'message': str, 'train_loss': float, 'samples': int}
    """
    try:
        corrections = get_corrections()
        if len(corrections) < 5:
            return {
                'success': False,
                'message': f'Недостаточно коррекций: {len(corrections)}/5 минимум. '
                           f'Соберите ещё {5 - len(corrections)} коррекций.',
                'train_loss': 0.0,
                'samples': len(corrections),
            }

        logger.info(f"=== ДООБУЧЕНИЕ МОДЕЛИ СЕНТИМЕНТА ===")
        logger.info(f"Коррекций: {len(corrections)}")
        logger.info(f"Параметры: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

        # Импортируем transformers
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import torch
        from torch.utils.data import Dataset

        # Загружаем модель
        model_path = BASE_MODEL_DIR / BASE_MODEL_NAME.replace('/', '_')
        if model_path.exists() and any(model_path.iterdir()):
            logger.info(f"Загрузка базовой модели из: {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            logger.info(f"Загрузка базовой модели из HuggingFace: {BASE_MODEL_NAME}")
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        # Подготовка данных
        texts = []
        labels = []
        for c in corrections:
            text = c.get('title', '')
            summary = c.get('summary', '')
            if summary:
                text = text + ' ' + summary
            text = text[:512]  # ограничение токенизатора

            corrected = c.get('corrected_label', 'NEUTRAL')
            label_lower = LABEL_MAP.get(corrected, 'neutral')

            # Маппинг в ID лейблов модели
            # rubert-tiny2-russian-financial-sentiment: 0=negative, 1=neutral, 2=positive
            label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
            label_id = label_to_id.get(label_lower, 1)

            texts.append(text)
            labels.append(label_id)

        logger.info(f"Подготовлено {len(texts)} примеров для обучения")

        # Датасет
        class SentimentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt',
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                }

        dataset = SentimentDataset(texts, labels, tokenizer)

        # Параметры обучения (CPU-оптимизированные)
        FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(FINETUNED_MODEL_DIR),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=max(1, len(texts) // 10),
            weight_decay=0.01,
            save_strategy='no',  # не сохраняем промежуточные
            logging_steps=max(1, len(texts) // (batch_size * 2)),
            report_to='none',  # отключаем wandb
            disable_tqdm=False,
            fp16=False,  # CPU → без fp16
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        # Обучение
        start_time = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start_time

        # Сохранение дообученной модели
        model.save_pretrained(str(FINETUNED_MODEL_DIR))
        tokenizer.save_pretrained(str(FINETUNED_MODEL_DIR))

        train_loss = train_result.training_loss

        logger.info(f"✅ Дообучение завершено за {elapsed:.0f}с")
        logger.info(f"   Loss: {train_loss:.4f}")
        logger.info(f"   Модель сохранена: {FINETUNED_MODEL_DIR}")

        return {
            'success': True,
            'message': f'Модель дообучена на {len(texts)} примерах. '
                       f'Loss: {train_loss:.4f}, время: {elapsed:.0f}с. '
                       f'Перезапустите систему для применения.',
            'train_loss': train_loss,
            'samples': len(texts),
            'elapsed_seconds': elapsed,
        }

    except Exception as e:
        logger.error(f"Ошибка дообучения: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'message': f'Ошибка дообучения: {e}',
            'train_loss': 0.0,
            'samples': 0,
        }


def is_finetuned_available() -> bool:
    """Проверка, есть ли дообученная модель."""
    return FINETUNED_MODEL_DIR.exists() and any(FINETUNED_MODEL_DIR.iterdir())


def get_finetuned_model_path() -> Optional[str]:
    """Путь к дообученной модели (или None)."""
    if is_finetuned_available():
        return str(FINETUNED_MODEL_DIR)
    return None


def get_stats() -> Dict:
    """Статистика по коррекциям и модели."""
    corrections = get_corrections()
    by_label = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    for c in corrections:
        label = c.get('corrected_label', 'NEUTRAL')
        by_label[label] = by_label.get(label, 0) + 1

    return {
        'total_corrections': len(corrections),
        'by_label': by_label,
        'finetuned_available': is_finetuned_available(),
        'finetuned_path': str(FINETUNED_MODEL_DIR) if is_finetuned_available() else None,
        'min_for_finetune': 5,
    }
